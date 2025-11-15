import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel as C
    from sklearn.gaussian_process.kernels import Matern
except ImportError as exc:
    raise ImportError(
        "baseline_final_search.py requires scikit-learn (sklearn). "
        "Install it via `pip install scikit-learn` before running the search."
    ) from exc

from baseline_final import (
    CFG,
    CrossAttnVLM,
    collate_fn,
    export_predictions,
    make_loader,
    resolve_dir_pairs,
    seed_everything,
    train_one_epoch,
)


SEARCH_SPACE = {
    "dim": [512, 1024],
    "cross_attn_dim": [512, 768, 1024],
    "cross_attn_layers": [1, 2, 4],
    "cross_attn_heads": [1, 2, 4, 8],
}


@dataclass
class IterationResult:
    iteration: int
    params: Dict[str, int]
    val_miou: float
    train_miou: float
    train_loss: float
    elapsed_sec: float


@dataclass
class DataContext:
    train_dataset: Dataset
    val_dataset: Dataset
    vocab: Any


class ParameterEncoder:
    """Normalizes discrete hyper-parameters into [0, 1] for GP fitting."""

    def __init__(self, search_space: Dict[str, Sequence[int]]):
        self.keys: Tuple[str, ...] = (
            "dim",
            "cross_attn_dim",
            "cross_attn_layers",
            "cross_attn_heads",
        )
        self.bounds: Dict[str, Tuple[float, float]] = {}
        for key in self.keys:
            values = list(search_space[key])
            lo = float(min(values))
            hi = float(max(values))
            self.bounds[key] = (lo, hi)

    def encode(self, params: Dict[str, int]) -> np.ndarray:
        vec: List[float] = []
        for key in self.keys:
            lo, hi = self.bounds[key]
            val = float(params[key])
            if math.isclose(hi, lo):
                vec.append(0.0)
            else:
                vec.append((val - lo) / (hi - lo))
        return np.asarray(vec, dtype=np.float64)


def expected_improvement(mean: float, std: float, best: float, xi: float) -> float:
    """Computes EI for maximization problems."""
    if std <= 1e-9:
        return 0.0
    diff = mean - best - xi
    z = diff / std
    cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    pdf = (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * z * z)
    ei = diff * cdf + std * pdf
    return max(0.0, float(ei))


class DiscreteBayesOptimizer:
    def __init__(
        self,
        search_space: Dict[str, Sequence[int]],
        seed: int,
        init_random: int,
        exploration: float,
    ):
        self.encoder = ParameterEncoder(search_space)
        self.candidates: List[Dict[str, int]] = self._build_candidates(search_space)
        self.remaining: set[int] = set(range(len(self.candidates)))
        self.rng = random.Random(seed)
        self.X: List[np.ndarray] = []
        self.y: List[float] = []
        self.gp: Optional[GaussianProcessRegressor] = None
        self.init_random = max(1, min(init_random, len(self.candidates)))
        self.exploration = exploration

    @staticmethod
    def _build_candidates(space: Dict[str, Sequence[int]]) -> List[Dict[str, int]]:
        combos: List[Dict[str, int]] = []
        for dim in space["dim"]:
            for cross_dim in space["cross_attn_dim"]:
                for layers in space["cross_attn_layers"]:
                    for heads in space["cross_attn_heads"]:
                        combos.append(
                            {
                                "dim": int(dim),
                                "cross_attn_dim": int(cross_dim),
                                "cross_attn_layers": int(layers),
                                "cross_attn_heads": int(heads),
                            }
                        )
        return combos

    def next_params(self) -> Optional[Dict[str, int]]:
        if not self.remaining:
            return None
        remaining_indices = tuple(self.remaining)
        if len(self.X) < self.init_random or self.gp is None:
            idx = self.rng.choice(remaining_indices)
        else:
            idx = self._select_with_ei(remaining_indices)
        self.remaining.remove(idx)
        return self.candidates[idx]

    def _select_with_ei(self, candidate_indices: Sequence[int]) -> int:
        assert self.gp is not None
        encoded = np.vstack([self.encoder.encode(self.candidates[i]) for i in candidate_indices])
        preds, stds = self.gp.predict(encoded, return_std=True)
        best_so_far = max(self.y)
        best_idx = None
        best_ei = -float("inf")
        for local_idx, global_idx in enumerate(candidate_indices):
            ei = expected_improvement(preds[local_idx], stds[local_idx], best_so_far, self.exploration)
            if ei > best_ei:
                best_ei = ei
                best_idx = global_idx
        if best_idx is None:
            # Fall back to RNG if EI collapses (e.g., identical std values).
            return self.rng.choice(candidate_indices)
        return best_idx

    def register(self, params: Dict[str, int], score: float):
        self.X.append(self.encoder.encode(params))
        self.y.append(float(score))
        self._fit_gp()

    def _fit_gp(self):
        if len(self.y) < 2:
            self.gp = None
            return
        try:
            X = np.vstack(self.X)
            y = np.asarray(self.y)
            kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=np.ones(X.shape[1]), nu=2.5)
            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-4,
                n_restarts_optimizer=3,
                normalize_y=True,
                random_state=self.rng.randint(0, 10**6),
            )
            gp.fit(X, y)
            self.gp = gp
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[warn] GP fitting failed: {exc}")
            self.gp = None


def build_loader(dataset: Dataset, args: argparse.Namespace, shuffle: bool, device: torch.device) -> DataLoader:
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": shuffle,
        "num_workers": args.num_workers,
        "collate_fn": collate_fn,
        "pin_memory": device.type == "cuda",
        "persistent_workers": False,
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 4
    return DataLoader(dataset, **loader_kwargs)


def evaluate_once(
    params: Dict[str, int],
    ctx: DataContext,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[float, float, float]:
    train_loader = build_loader(ctx.train_dataset, args, shuffle=True, device=device)
    val_loader = build_loader(ctx.val_dataset, args, shuffle=False, device=device)

    model = CrossAttnVLM(
        vocab_size=len(ctx.vocab.itos),
        dim=params["dim"],
        pretrained_backbone=not args.no_pretrain,
        img_size=args.img_size,
        cross_attn_dim=params["cross_attn_dim"],
        cross_attn_layers=params["cross_attn_layers"],
        cross_attn_heads=params["cross_attn_heads"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    train_loss, train_miou = train_one_epoch(
        model,
        train_loader,
        optimizer,
        scaler,
        device,
        desc="search-train",
        on_half_epoch=None,
    )
    val_miou = export_predictions(
        model,
        val_loader,
        out_csv="",
        device=device,
        compute_iou=True,
        progress_desc="search-val",
    )

    # Cleanup references so CUDA memory can be reclaimed quickly.
    del model
    del optimizer
    del scaler
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return (
        float(val_miou) if val_miou is not None else 0.0,
        float(train_miou),
        float(train_loss),
    )


def prepare_datasets(args: argparse.Namespace) -> DataContext:
    train_pairs = resolve_dir_pairs(
        getattr(args, "train_json_dir", None),
        getattr(args, "train_jpg_dir", None),
        getattr(args, "train_root", None),
        "train",
    )
    val_pairs = resolve_dir_pairs(
        getattr(args, "val_json_dir", None),
        getattr(args, "val_jpg_dir", None),
        getattr(args, "val_root", None),
        "val",
    )

    train_ds, _, vocab = make_loader(
        train_pairs,
        vocab=None,
        build_vocab=True,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=0,
        shuffle=True,
    )
    val_ds, _, _ = make_loader(
        val_pairs,
        vocab=vocab,
        build_vocab=False,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=0,
        shuffle=False,
    )

    print(f"[data] train samples={len(train_ds)}  val samples={len(val_ds)}")
    return DataContext(train_dataset=train_ds, val_dataset=val_ds, vocab=vocab)


def save_results(path: Optional[str], payload: Dict[str, Any]):
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[search] saved results to {path}")


def run_search(args: argparse.Namespace):
    seed_everything(args.seed)
    device = torch.device(args.device if args.device in {"cuda", "cpu"} else "cuda")
    if device.type == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")

    ctx = prepare_datasets(args)
    optimizer = DiscreteBayesOptimizer(
        search_space=SEARCH_SPACE,
        seed=args.seed,
        init_random=min(args.init_random, args.iterations),
        exploration=args.ei_exploration,
    )

    results: List[IterationResult] = []
    best_result: Optional[IterationResult] = None

    for itr in range(1, args.iterations + 1):
        params = optimizer.next_params()
        if params is None:
            print("[search] no more unique configurations left to evaluate.")
            break

        start = time.time()
        try:
            val_miou, train_miou, train_loss = evaluate_once(params, ctx, args, device)
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"[search] iteration {itr} failed with error: {exc}")
            val_miou, train_miou, train_loss = 0.0, 0.0, float("inf")

        elapsed = time.time() - start
        iteration_result = IterationResult(
            iteration=itr,
            params=params,
            val_miou=val_miou,
            train_miou=train_miou,
            train_loss=train_loss,
            elapsed_sec=elapsed,
        )
        results.append(iteration_result)
        optimizer.register(params, val_miou)

        if best_result is None or val_miou > best_result.val_miou:
            best_result = iteration_result

        print(
            f"[search] iter={itr}/{args.iterations} "
            f"params={params} val_mIoU={val_miou:.4f} "
            f"train_mIoU={train_miou:.4f} loss={train_loss:.4f} "
            f"elapsed={elapsed/60:.2f}m "
            f"best={best_result.val_miou:.4f}"
        )

    payload = {
        "search_space": SEARCH_SPACE,
        "config": vars(args),
        "results": [asdict(r) for r in results],
        "best": asdict(best_result) if best_result else None,
    }
    save_results(args.results_path, payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bayesian hyper-parameter search for baseline_final Cross-Attention VLM."
    )
    parser.add_argument("--iterations", type=int, default=10, help="Total BO iterations.")
    parser.add_argument(
        "--init_random",
        type=int,
        default=3,
        help="Number of initial random samples before fitting the GP model.",
    )
    parser.add_argument(
        "--ei_exploration",
        type=float,
        default=0.01,
        help="Exploration parameter (xi) for Expected Improvement acquisition.",
    )
    parser.add_argument("--batch_size", type=int, default=CFG.BATCH_SIZE)
    parser.add_argument("--img_size", type=int, default=CFG.IMG_SIZE)
    parser.add_argument("--lr", type=float, default=CFG.LEARNING_RATE)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=CFG.NUM_WORKERS)
    parser.add_argument("--no_pretrain", action="store_true")
    parser.add_argument("--seed", type=int, default=CFG.SEED)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default="./outputs_baseline/bo/baseline_final_search_results.json",
    )
    parser.add_argument("--train_root", type=str, default=CFG.TRAIN_ROOT)
    parser.add_argument("--train_json_dir", type=str, default=CFG.TRAIN_JSON_DIR)
    parser.add_argument("--train_jpg_dir", type=str, default=CFG.TRAIN_JPG_DIR)
    parser.add_argument("--val_root", type=str, default=CFG.VAL_ROOT)
    parser.add_argument("--val_json_dir", type=str, default=CFG.VAL_JSON_DIR)
    parser.add_argument("--val_jpg_dir", type=str, default=CFG.VAL_JPG_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    args.iterations = max(1, args.iterations)
    run_search(args)


if __name__ == "__main__":
    main()
