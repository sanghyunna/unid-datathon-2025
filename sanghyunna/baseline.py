import os
import io
import json
import math
import time
import random
import argparse
import zipfile
from glob import glob
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader



# torchvision 백본 사용 가능 여부 체크
try:
    from torchvision.models import resnet50, ResNet50_Weights
    _BACKBONE_OK = True
except Exception:
    _BACKBONE_OK = False

class CFG:
    # Core
    IMG_SIZE: int = 512
    EPOCHS: int = 3
    LEARNING_RATE: float = 1e-4
    BATCH_SIZE: int = 64
    SEED: int = 42
    DIM: int = 512
    NUM_WORKERS: int = 14
    NO_PRETRAIN: bool = False  # True → disable ImageNet weights

    # Paths (override by CLI if desired)
    CKPT_PATH: str = "./outputs/ckpt/cross_attn_vlm.pth"
    RESUME_CKPT_PATH: str = "./outputs/ckpt/cross_attn_vlm_ep10.0.pth"
    EVAL_CSV: str = "./outputs/preds/eval_pred.csv"
    PRED_CSV: str = "./outputs/preds/test_pred.csv"
    SUBMISSION_ZIP: str = "./outputs/submission.zip"

    # Optional split-specific overrides
    TRAIN_ROOT: Optional[str] = "../data/train_valid/train"
    VAL_ROOT: Optional[str] = "../data/train_valid/valid"
    TEST_ROOT: Optional[str] = "../data/open/test"
    TRAIN_JSON_DIR: Optional[str] = None
    TRAIN_JPG_DIR: Optional[str] = None
    VAL_JSON_DIR: Optional[str] = None
    VAL_JPG_DIR: Optional[str] = None
    TEST_JSON_DIR: Optional[str] = None
    TEST_JPG_DIR: Optional[str] = None

BAD_JSON_WHITELIST_FILE = "./bad_json_whitelist.txt"


def load_bad_json_whitelist(path: str) -> set:
    """bad_json_whitelist.txt를 읽어서 스킵할 JSON 경로 set으로 반환."""
    if not os.path.isfile(path):
        # 파일이 아예 없으면 빈 집합
        return set()

    bad = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # 빈 줄 / 주석 무시
            if not line or line.startswith("#"):
                continue
            bad.add(os.path.normpath(line))
    print(f"[bad-json] loaded {len(bad)} entries from {path}")
    return bad


BAD_JSON_WHITELIST: set = load_bad_json_whitelist(BAD_JSON_WHITELIST_FILE)


def append_bad_json_whitelist(path: str) -> bool:
    norm = os.path.normpath(path)
    if norm in BAD_JSON_WHITELIST:
        return False
    with open(BAD_JSON_WHITELIST_FILE, "a", encoding="utf-8") as f:
        f.write(f"{norm}\n")
    BAD_JSON_WHITELIST.add(norm)
    return True


def handle_bad_json_exception(path: str, exc: Exception) -> bool:
    try:
        size = os.path.getsize(path)
    except OSError:
        return False

    if size == 0:
        added = append_bad_json_whitelist(path)
        status = "added" if added else "exists"
        print(f"[bad-json] zero-byte file detected ({status} in whitelist): {path}")
        return True
    return False


def require_path(path: Optional[str], flag: str) -> str:
    if not path:
        raise ValueError(f"{flag} must be provided via CLI or CFG.")
    return path


def discover_json_jpg_pairs(root_dir: str) -> List[Tuple[str, str]]:
    if not root_dir:
        raise ValueError("root_dir must be provided for auto-discovery")
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Root directory not found: {root_dir}")

    child_dirs = sorted(
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    )

    suffixes = [
        "_json", "json", "_jpg", "jpg", "_jpeg", "jpeg",
        "_img", "img", "_image", "image", "_images", "images",
        "_query", "query", "_queries", "queries"
    ]

    def normalize_base(name: str) -> str:
        lower = name.lower()
        for suf in suffixes:
            if lower.endswith(suf):
                lower = lower[: -len(suf)]
                break
        lower = lower.rstrip("_-")
        return lower or "root"

    def has_files(path: str, patterns: List[str]) -> bool:
        for patt in patterns:
            if glob(os.path.join(path, patt)):
                return True
        return False

    def is_json_dir(name: str, path: str) -> bool:
        lower = name.lower()
        if any(tag in lower for tag in ["json", "query", "annotation"]):
            return True
        return has_files(path, ["*.json"])

    def is_image_dir(name: str, path: str) -> bool:
        lower = name.lower()
        if any(tag in lower for tag in ["jpg", "jpeg", "img", "image", "images", "photo"]):
            return True
        return has_files(path, ["*.jpg", "*.jpeg", "*.png"])

    json_dirs: List[str] = []
    img_dirs: List[str] = []
    buckets: Dict[str, Dict[str, Optional[str]]] = {}

    for name in child_dirs:
        path = os.path.join(root_dir, name)
        json_like = is_json_dir(name, path)
        img_like = is_image_dir(name, path)
        if not json_like and not img_like:
            continue
        base = normalize_base(name)
        entry = buckets.setdefault(base, {"json": None, "jpg": None})
        if json_like and entry["json"] is None:
            entry["json"] = path
            json_dirs.append(path)
        if img_like and entry["jpg"] is None:
            entry["jpg"] = path
            img_dirs.append(path)

    pairs = [(info["json"], info["jpg"]) for info in buckets.values() if info["json"] and info["jpg"]]

    if not pairs and len(json_dirs) == 1 and len(img_dirs) == 1:
        pairs = [(json_dirs[0], img_dirs[0])]

    if not pairs:
        raise ValueError(f"Could not auto-match JSON/JPG directories under {root_dir}")
    return pairs


def resolve_dir_pairs(json_dir: Optional[str], jpg_dir: Optional[str], root_dir: Optional[str], split_name: str) -> List[Tuple[str, str]]:
    if root_dir:
        return discover_json_jpg_pairs(root_dir)
    if json_dir and jpg_dir:
        return [(json_dir, jpg_dir)]
    if json_dir or jpg_dir:
        raise ValueError(
            f"Split '{split_name}' requires both JSON and JPG directories when a root is not provided."
        )
    raise ValueError(
        f"Provide either --{split_name}_root or both --{split_name}_json_dir/--{split_name}_jpg_dir."
    )

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def find_jsons(json_dir: str) -> List[str]:
    if os.path.isdir(json_dir):
        return sorted(glob(os.path.join(json_dir, "*.json")))
    raise FileNotFoundError(f"json_dir not found: {json_dir}")


def read_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        # Make it obvious which file is broken
        raise ValueError(f"JSON decode error in file: {path}\n{e}") from e



def get_image_path(json_path: str, data: Dict[str, Any], jpg_dir: Optional[str] = None) -> str:
    # Prefer explicit mapping via source_data_name_jpg
    src = data.get("source_data_info", {})
    jpg_name = src.get("source_data_name_jpg", None)
    if jpg_dir and jpg_name:
        path = os.path.join(jpg_dir, jpg_name)
        if os.path.exists(path):
            return path
    # Fallback: .../json/... -> .../jpg/...
    if jpg_name:
        maybe = json_path.replace(os.sep + "json" + os.sep, os.sep + "jpg" + os.sep)
        maybe = os.path.join(os.path.dirname(maybe), jpg_name) if os.path.isdir(os.path.dirname(maybe)) else maybe
        if os.path.exists(maybe):
            return maybe
    # Last resort: same dir, MI3 -> MI2.jpg
    base = os.path.splitext(os.path.basename(json_path))[0]
    sibling = os.path.join(os.path.dirname(json_path), base.replace("MI3", "MI2") + ".jpg")
    if os.path.exists(sibling):
        return sibling
    raise FileNotFoundError(f"Could not resolve JPG for {json_path} (jpg_dir={jpg_dir})")


def simple_tokenize(s: str) -> List[str]:
    s = (s or "")
    s = s.replace("##", " ").replace(",", " ").replace("(", " ").replace(")", " ")
    s = s.replace(":", " ").replace("?", " ").replace("!", " ").replace("·", " ")
    return [t for t in s.strip().split() if t]


def is_visual_ann(a: dict) -> bool:
    """Use only visual elements that have a query: V* class_id or table/chart-related class_name, and non-empty instruction."""
    cid = str(a.get("class_id", "") or "")
    cname = str(a.get("class_name", "") or "")
    has_q = bool(str(a.get("visual_instruction", "") or "").strip())
    looks_visual = cid.startswith("V") or any(k in cname for k in ["표", "차트", "그래프", "chart", "table"])
    return has_q and looks_visual

class Vocab:
    def __init__(self, min_freq: int = 1):
        self.min_freq = min_freq
        self.freq: Dict[str, int] = {}
        self.itos: List[str] = ["<pad>", "<unk>"]
        self.stoi: Dict[str, int] = {tok: i for i, tok in enumerate(self.itos)}

    def build(self, texts: List[str]):
        for s in texts:
            for tok in simple_tokenize(s):
                self.freq[tok] = self.freq.get(tok, 0) + 1
        for tok, f in sorted(self.freq.items(), key=lambda x: (-x[1], x[0])):
            if f >= self.min_freq and tok not in self.stoi:
                self.stoi[tok] = len(self.itos)
                self.itos.append(tok)

    def encode(self, s: str, max_len: int = 40) -> List[int]:
        toks = simple_tokenize(s)[:max_len]
        if not toks:
            return [1]  # ensure length>=1 with <unk>
        return [self.stoi.get(t, 1) for t in toks]


class UniDSet(Dataset):
    def __init__(self, json_files: List[str], jpg_dir: Optional[str] = None, vocab: Optional[Vocab] = None,
                 build_vocab: bool = False, resize_to: Tuple[int, int] = (CFG.IMG_SIZE, CFG.IMG_SIZE),
                 supervised_only: bool = False):
        self.vocab = vocab if vocab is not None else Vocab(min_freq=1)
        self.items: List[Dict[str, Any]] = []

        # 이미지 리사이즈/감독 여부 및 캐시 파일 suffix 설정
        self.resize_to = resize_to
        self.supervised_only = supervised_only
        self.cache_suffix = f"_cache_{resize_to[0]}x{resize_to[1]}.pt"

        desc_name = os.path.basename(jpg_dir) if jpg_dir else (
            os.path.basename(os.path.dirname(json_files[0])) if json_files else "jsons"
        )

        for jf in tqdm(json_files, desc=f"[data] {desc_name}", leave=False, dynamic_ncols=True):
            # 0) 화이트리스트에 있는 JSON은 스킵
            jf_norm = os.path.normpath(jf)
            if jf_norm in BAD_JSON_WHITELIST:
                print(f"[bad-json] skipping whitelisted JSON: {jf}")
                continue

            # 1) JSON 읽기
            try:
                data = read_json(jf)
            except ValueError as err:
                if handle_bad_json_exception(jf, err):
                    continue
                raise

            ann = data.get("learning_data_info", {}).get("annotation", [])
            img_path = get_image_path(jf, data, jpg_dir=jpg_dir)

            for a in ann:
                if not is_visual_ann(a):
                    continue

                qid = a.get("instance_id", "")
                qtxt = str(a.get("visual_instruction", "")).strip()
                bbox = a.get("bounding_box", None)  # train/val에는 bbox, test는 없을 수 있음

                if supervised_only and not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
                    continue

                cname = a.get("class_name", "")

                # 캐시 파일 경로: 원본 이미지와 동일 경로, 다른 확장자(.pt)
                base, _ = os.path.splitext(img_path)
                cache_path = base + self.cache_suffix

                self.items.append({
                    "json": jf,
                    "img": img_path,
                    "query_id": qid,
                    "query": qtxt,
                    "bbox": bbox,
                    "class_name": cname,
                    "cache_path": cache_path,
                })

        if build_vocab:
            self.vocab.build([it["query"] for it in self.items])

        for it in self.items:
            it["encoded_query"] = self.vocab.encode(it["query"], max_len=40)

        if _BACKBONE_OK:
            from torchvision import transforms as T
            self.tf = T.Compose([T.Resize(resize_to), T.ToTensor()])
        else:
            self.tf = None  # will manually convert

    def __len__(self):
        return len(self.items)

    @staticmethod
    def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
        arr = np.array(img).astype(np.float32) / 255.0
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr)

    @staticmethod
    def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
        arr = np.array(img).astype(np.float32) / 255.0
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        it = self.items[idx]
        cache_path = it.get("cache_path", None)

        # =========================
        # 1) 캐시에서 먼저 시도
        # =========================
        img_t: Optional[torch.Tensor] = None
        W: int
        H: int

        if cache_path is not None and os.path.isfile(cache_path):
            try:
                # NOTE: PyTorch 2.6 대비 weights_only=False 명시
                cached = torch.load(cache_path, map_location="cpu", weights_only=False)
                # {"image": Tensor, "orig_size": (W, H)} 포맷을 기대
                if isinstance(cached, dict) and "image" in cached and "orig_size" in cached:
                    img_t = cached["image"]
                    W, H = cached["orig_size"]
                else:
                    raise ValueError("unexpected cache format")
            except Exception as e:
                print(f"[cache] failed to load {cache_path}: {e}")
                img_t = None

        # ==========================================
        # 2) 캐시가 없거나 실패하면 원본에서 생성
        # ==========================================
        if img_t is None:
            img = Image.open(it["img"]).convert("RGB")
            W, H = img.size

            if self.tf is not None:
                img_t = self.tf(img)  # Resize + ToTensor
            else:
                img = img.resize(self.resize_to, Image.BILINEAR)
                img_t = self._pil_to_tensor(img)

            # 생성한 결과를 캐시에 저장
            if cache_path is not None and not os.path.exists(cache_path):
                try:
                    torch.save({"image": img_t, "orig_size": (W, H)}, cache_path)
                except Exception as e:
                    print(f"[cache] failed to save {cache_path}: {e}")

        # =========================
        # 3) 나머지 메타/타깃 구성
        # =========================
        ids = it["encoded_query"]
        length = max(1, len(ids))  # safety: ensure >=1

        sample: Dict[str, Any] = {
            "image": img_t,
            "query_ids": torch.tensor(ids, dtype=torch.long),
            "length": torch.tensor(length, dtype=torch.long),
            "query_text": it["query"],
            "query_id": it["query_id"],
            "orig_size": (W, H),
            "class_name": it["class_name"],
        }

        # JSON bbox (x, y, w, h) in pixel → (cx, cy, w, h) normalized
        if it["bbox"] is not None and isinstance(it["bbox"], (list, tuple)) and len(it["bbox"]) == 4:
            x, y, w, h = it["bbox"]
            cx = (x + w / 2.0) / W
            cy = (y + h / 2.0) / H
            nw = w / W
            nh = h / H
            target = torch.tensor([cx, cy, nw, nh], dtype=torch.float32)
        else:
            target = None

        sample["target"] = target
        return sample



def collate_fn(batch: List[Dict[str, Any]]):
    # pad variable-length queries
    max_len = max(max(1, int(b["length"])) for b in batch)
    B = len(batch)
    ids = torch.zeros(B, max_len, dtype=torch.long)
    lens = torch.zeros(B, dtype=torch.long)
    imgs = torch.stack([b["image"] for b in batch], dim=0)
    targets = []
    meta = []
    for i, b in enumerate(batch):
        l = max(1, int(b["length"]))
        ids[i, :l] = b["query_ids"][:l]
        lens[i] = l
        targets.append(b["target"])
        meta.append({
            "query_id": b["query_id"],
            "query_text": b["query_text"],
            "orig_size": b["orig_size"],
            "class_name": b["class_name"],
        })

    stacked_targets = None
    if all(t is not None for t in targets) and len(targets) > 0:
        stacked_targets = torch.stack(targets, dim=0)
    return imgs, ids, lens, targets, meta, stacked_targets

class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = CFG.DIM, hidden: int = CFG.DIM):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.gru = nn.GRU(input_size=emb_dim, hidden_size=hidden, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden * 2, emb_dim)

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = self.emb(tokens)  # (B, L, E)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, h = self.gru(packed)
        h_cat = torch.cat([h[-2], h[-1]], dim=-1)  # (B, 2*hidden)
        q = self.proj(h_cat)  # (B, D)
        return q


class TinyCNN(nn.Module):
    def __init__(self, out_dim: int = CFG.DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, out_dim, 3, stride=2, padding=1), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)  # (B, D, H', W')


class ImageEncoder(nn.Module):
    def __init__(self, out_dim: int = CFG.DIM, pretrained: bool = True, img_size: int = CFG.IMG_SIZE):
        super().__init__()
        self.resize = None
        if _BACKBONE_OK:
            try:
                weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
                m = resnet50(weights=weights)
                layers = list(m.children())[:-2]  # (B, 512, H/32, W/32)
                self.backbone = nn.Sequential(*layers)
                self.proj = nn.Conv2d(2048, out_dim, 1)
                from torchvision import transforms as T
                self.resize = T.Compose([T.Resize((img_size, img_size)), T.ToTensor()])
            except Exception:
                self.backbone = TinyCNN(out_dim)
                self.proj = nn.Identity()
        else:
            self.backbone = TinyCNN(out_dim)
            self.proj = nn.Identity()

    def forward(self, x):
        f = self.backbone(x)
        f = self.proj(f)
        return f


class CrossAttentionBBox(nn.Module):
    def __init__(self, dim: int = CFG.DIM):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Conv2d(dim, dim, 1)
        self.v_proj = nn.Conv2d(dim, dim, 1)
        self.bbox_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, 4)  # (cx, cy, w, h) normalized via sigmoid
        )

    def forward(self, q_vec: torch.Tensor, fmap: torch.Tensor) -> torch.Tensor:
        B, D, H, W = fmap.shape
        q = self.q_proj(q_vec)             # (B, D)
        K = self.k_proj(fmap)              # (B, D, H, W)
        V = self.v_proj(fmap)              # (B, D, H, W)

        Kf = K.flatten(2).transpose(1, 2)  # (B, HW, D)
        Vf = V.flatten(2).transpose(1, 2)  # (B, HW, D)
        q = q.unsqueeze(1)                 # (B, 1, D)

        attn = torch.matmul(q, Kf.transpose(1, 2)) / math.sqrt(D)  # (B, 1, HW)
        attn = torch.softmax(attn, dim=-1)
        ctx = torch.matmul(attn, Vf).squeeze(1)  # (B, D)

        pred = self.bbox_head(ctx)         # (B, 4)
        pred = torch.sigmoid(pred)         # normalize to [0,1]
        return pred


class CrossAttnVLM(nn.Module):
    def __init__(self, vocab_size: int, dim: int = CFG.DIM, pretrained_backbone: bool = True, img_size: int = CFG.IMG_SIZE):
        super().__init__()
        self.txt = TextEncoder(vocab_size=vocab_size, emb_dim=dim, hidden=dim)
        self.img = ImageEncoder(out_dim=dim, pretrained=pretrained_backbone, img_size=img_size)
        self.head = CrossAttentionBBox(dim=dim)

    def forward(self, images: torch.Tensor, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        q = self.txt(tokens, lengths)             # (B, D)
        fmap = self.img(images)                   # (B, D, H', W')
        pred_norm = self.head(q, fmap)            # (B, 4) in [0,1]
        return pred_norm

def iou_xywh_pixel(pred_xywh, gt_xywh):
    px, py, pw, ph = pred_xywh
    gx, gy, gw, gh = gt_xywh
    px2, py2 = px + pw, py + ph
    gx2, gy2 = gx + gw, gy + gh
    ix1, iy1 = max(px, gx), max(py, gy)
    ix2, iy2 = min(px2, gx2), min(py2, gy2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = pw * ph + gw * gh - inter if (pw * ph + gw * gh - inter) > 0 else 1e-6
    return inter / union

def batch_iou_norm(pred: torch.Tensor, tgt: torch.Tensor) -> float:
    """
    pred, tgt: (B,4) in normalized (cx, cy, w, h) \in [0,1]
    IoU는 스케일 불변이라, 이 좌표계에서 계산한 값이 픽셀 기준 IoU와 동일합니다.
    """
    # (cx, cy, w, h) -> (x1, y1, x2, y2)
    px1 = pred[:, 0] - pred[:, 2] / 2.0
    py1 = pred[:, 1] - pred[:, 3] / 2.0
    px2 = pred[:, 0] + pred[:, 2] / 2.0
    py2 = pred[:, 1] + pred[:, 3] / 2.0

    gx1 = tgt[:, 0] - tgt[:, 2] / 2.0
    gy1 = tgt[:, 1] - tgt[:, 3] / 2.0
    gx2 = tgt[:, 0] + tgt[:, 2] / 2.0
    gy2 = tgt[:, 1] + tgt[:, 3] / 2.0

    ix1 = torch.max(px1, gx1)
    iy1 = torch.max(py1, gy1)
    ix2 = torch.min(px2, gx2)
    iy2 = torch.min(py2, gy2)

    inter_w = (ix2 - ix1).clamp(min=0)
    inter_h = (iy2 - iy1).clamp(min=0)
    inter = inter_w * inter_h

    area_p = (px2 - px1).clamp(min=0) * (py2 - py1).clamp(min=0)
    area_g = (gx2 - gx1).clamp(min=0) * (gy2 - gy1).clamp(min=0)
    union = (area_p + area_g - inter).clamp(min=1e-6)

    iou = inter / union
    return float(iou.mean().item())


def export_predictions(model: nn.Module, loader: DataLoader, out_csv: str,
                       device: torch.device, compute_iou: bool = False,
                       progress_desc: str = "predict") -> Optional[float]:
    rows = []
    ious: List[float] = []

    # NEW: 원래 training 상태 저장
    was_training = model.training
    model.eval()

    with torch.no_grad():
        iterator = tqdm(loader, desc=progress_desc, leave=False, dynamic_ncols=True)
        for imgs, ids, lens, targets, meta, _ in iterator:
            imgs = imgs.to(device, non_blocking=True)
            ids = ids.to(device, non_blocking=True)
            lens = lens.to(device, non_blocking=True)
            pred = model(imgs, ids, lens)
            for i in range(imgs.size(0)):
                W, H = meta[i]["orig_size"]
                cx, cy, nw, nh = [float(v) for v in pred[i].cpu().numpy().tolist()]
                x = (cx - nw / 2.0) * W
                y = (cy - nh / 2.0) * H
                w = nw * W
                h = nh * H
                rows.append({
                    "query_id": meta[i]["query_id"],
                    "query_text": meta[i]["query_text"],
                    "pred_x": x,
                    "pred_y": y,
                    "pred_w": w,
                    "pred_h": h,
                })
                if compute_iou and targets[i] is not None:
                    gt = [float(v) for v in targets[i].numpy().tolist()]
                    gx = (gt[0] - gt[2] / 2.0) * W
                    gy = (gt[1] - gt[3] / 2.0) * H
                    gw = gt[2] * W
                    gh = gt[3] * H
                    ious.append(iou_xywh_pixel([x, y, w, h], [gx, gy, gw, gh]))

    if out_csv:
        out_dir = os.path.dirname(out_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        import pandas as pd
        df = pd.DataFrame(rows, columns=["query_id", "query_text", "pred_x", "pred_y", "pred_w", "pred_h"])
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"[Saved] {out_csv}")

    miou: Optional[float] = None
    if compute_iou and ious:
        miou = float(np.mean(ious))
    elif compute_iou:
        print("[Eval] No ground-truth boxes available; IoU skipped.")

    # NEW: 원래 상태로 복원
    if was_training:
        model.train()

    return miou



def make_loader(dir_pairs: List[Tuple[str, str]], vocab: Optional[Vocab] = None, build_vocab: bool = False,
                batch_size: int = CFG.BATCH_SIZE, img_size: int = CFG.IMG_SIZE,
                num_workers: int = CFG.NUM_WORKERS, shuffle: bool = False):
    if not dir_pairs:
        raise ValueError("dir_pairs must contain at least one (json_dir, jpg_dir) tuple")

    vocab = vocab if vocab is not None else Vocab(min_freq=1)
    datasets: List[Dataset] = []
    for json_dir, jpg_dir in dir_pairs:
        json_files = find_jsons(json_dir)
        subset = UniDSet(
            json_files,
            jpg_dir=jpg_dir,
            vocab=vocab,
            build_vocab=build_vocab,
            resize_to=(img_size, img_size),
            supervised_only=build_vocab,
        )
        if len(subset) == 0:
            print(f"[warn] No usable samples in {json_dir}; skipping")
            continue
        datasets.append(subset)

    if not datasets:
        raise RuntimeError("No datasets constructed from provided directories.")

    if len(datasets) == 1:
        full_ds: Dataset = datasets[0]
    else:
        full_ds = torch.utils.data.ConcatDataset(datasets)
        setattr(full_ds, "vocab", vocab)

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "collate_fn": collate_fn,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 8

    dl = DataLoader(full_ds, **loader_kwargs)
    return full_ds, dl, vocab


def save_checkpoint(model: nn.Module, vocab: Vocab, img_size: int, ckpt_path: str, dim: int, no_pretrain: bool):
    os.makedirs(os.path.dirname(ckpt_path) or '.', exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "vocab_itos": vocab.itos,
        "dim": dim,
        "no_pretrain": no_pretrain,
        "img_size": img_size,
    }, ckpt_path)
    print(f"[Saved] {ckpt_path}")


from typing import Tuple  # 이미 있으면 생략

def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                    scaler: torch.amp.GradScaler, device: torch.device,
                    desc: Optional[str] = None, on_half_epoch=None) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_iou = 0.0
    total = 0

    num_batches = len(loader)
    half_point = num_batches // 2  # 0.5 epoch 시점 (batch 기준)

    iterator = tqdm(loader, desc=desc or "train", leave=False, dynamic_ncols=True)
    for step, (imgs, ids, lens, targets, meta, stacked_targets) in enumerate(iterator, start=1):
        imgs = imgs.to(device, non_blocking=True)
        ids = ids.to(device, non_blocking=True)
        lens = lens.to(device, non_blocking=True)
        if stacked_targets is not None:
            t = stacked_targets.to(device, non_blocking=True)
        else:
            t = torch.stack([tar for tar in targets if tar is not None], dim=0).to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            pred = model(imgs, ids, lens)   # (B,4) normalized
            loss = F.smooth_l1_loss(pred, t, reduction="mean")

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        B = imgs.size(0)
        running_loss += float(loss.item()) * B
        total += B

        # NEW: batch 단위 IoU 누적 (normalized 좌표에서 계산)
        with torch.no_grad():
            batch_iou = batch_iou_norm(pred.detach(), t)
        running_iou += batch_iou * B

        # NEW: 0.5 epoch 지점에 도달하면 지금까지의 train mIoU 넘겨줌
        if on_half_epoch is not None and half_point > 0 and step == half_point:
            cur_train_iou = running_iou / max(1, total)
            on_half_epoch(cur_train_iou)

    avg_loss = running_loss / max(1, total)
    avg_iou = running_iou / max(1, total)
    return avg_loss, avg_iou



def evaluate_loss(model: nn.Module, loader: DataLoader, device: torch.device,
                  desc: Optional[str] = None) -> float:
    model.eval()
    running = 0.0
    total = 0
    with torch.no_grad():
        iterator = tqdm(loader, desc=desc or "eval", leave=False, dynamic_ncols=True)
        for imgs, ids, lens, targets, meta, stacked_targets in iterator:
            valid_idx = [i for i, tar in enumerate(targets) if tar is not None]
            if not valid_idx:
                continue
            imgs = imgs.to(device, non_blocking=True)
            ids = ids.to(device, non_blocking=True)
            lens = lens.to(device, non_blocking=True)
            pred = model(imgs, ids, lens)
            pred_sel = pred[valid_idx]
            if stacked_targets is not None:
                tgt_sel = stacked_targets[valid_idx].to(device, non_blocking=True)
            else:
                tgt_sel = torch.stack([targets[i] for i in valid_idx], dim=0).to(device)
            loss = F.smooth_l1_loss(pred_sel, tgt_sel, reduction="mean")
            running += float(loss.item()) * len(valid_idx)
            total += len(valid_idx)
    return running / max(1, total)


def train_loop(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_pairs = resolve_dir_pairs(
        getattr(args, "train_json_dir", None),
        getattr(args, "train_jpg_dir", None),
        getattr(args, "train_root", None),
        "train",
    )

    # ============================================
    # Resume 여부에 따라 model / vocab / loader 구성
    # ============================================
    resume_ckpt = getattr(args, "resume_ckpt", None)

    if resume_ckpt:
        # 기존 체크포인트에서 모델 + vocab + img_size 로드
        model, vocab, used_img_size = _load_model_from_ckpt(resume_ckpt, device)
        # 실제 dim은 모델의 텍스트 임베딩 차원에서 읽어온다
        ckpt_dim = model.txt.emb.embedding_dim

        # 기존 vocab을 그대로 사용 (build_vocab=False)
        train_ds, train_dl, _ = make_loader(
            train_pairs,
            vocab=vocab,
            build_vocab=False,
            batch_size=args.batch_size,
            img_size=used_img_size,
            num_workers=args.num_workers,
            shuffle=True,
        )
        print(f"[Resume] Loaded checkpoint from {resume_ckpt}")
    else:
        # 처음부터 학습
        used_img_size = args.img_size
        train_ds, train_dl, vocab = make_loader(
            train_pairs,
            vocab=None,
            build_vocab=True,
            batch_size=args.batch_size,
            img_size=used_img_size,
            num_workers=args.num_workers,
            shuffle=True,
        )
        model = CrossAttnVLM(
            vocab_size=len(vocab.itos),
            dim=args.dim,
            pretrained_backbone=not args.no_pretrain,
            img_size=used_img_size,
        ).to(device)
        ckpt_dim = args.dim

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    if hasattr(train_dl, "dataset") and hasattr(train_dl.dataset, "__len__"):
        total_samples = len(train_dl.dataset)  # type: ignore[arg-type]
    else:
        total_samples = len(train_dl) * args.batch_size

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        batch_iter = tqdm(
            train_dl,
            desc=f"train epoch {epoch}/{args.epochs}",
            leave=False,
            dynamic_ncols=True,
        )
        for imgs, ids, lens, targets, meta, stacked_targets in batch_iter:
            imgs = imgs.to(device, non_blocking=True)
            ids = ids.to(device, non_blocking=True)
            lens = lens.to(device, non_blocking=True)
            if stacked_targets is not None:
                t = stacked_targets.to(device, non_blocking=True)
            else:
                t = torch.stack([tar for tar in targets if tar is not None], dim=0).to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                pred = model(imgs, ids, lens)   # (B,4) normalized
                loss = F.smooth_l1_loss(pred, t, reduction="mean")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.item()) * imgs.size(0)

        scheduler.step()
        avg = running / total_samples
        print(f"[Epoch {epoch}/{args.epochs}] loss={avg:.4f}  lr={scheduler.get_last_lr()[0]:.6f}")

    # used_img_size / ckpt_dim 사용해서 저장 (dim 메타가 실제 모델과 일치하도록)
    save_checkpoint(model, vocab, used_img_size, args.save_ckpt, ckpt_dim, args.no_pretrain)



def _load_model_from_ckpt(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    vocab = Vocab(); vocab.itos = ckpt["vocab_itos"]; vocab.stoi = {t: i for i, t in enumerate(vocab.itos)}
    model = CrossAttnVLM(vocab_size=len(vocab.itos), dim=ckpt["dim"],
                         pretrained_backbone=not ckpt.get("no_pretrain", False),
                         img_size=ckpt.get("img_size", CFG.IMG_SIZE)).to(device)
    model.load_state_dict(ckpt["model_state"]); model.eval()
    img_size = ckpt.get("img_size", CFG.IMG_SIZE)
    return model, vocab, img_size


def evaluate_loop(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab, img_size = _load_model_from_ckpt(args.ckpt, device)

    val_pairs = resolve_dir_pairs(
        getattr(args, "val_json_dir", None),
        getattr(args, "val_jpg_dir", None),
        getattr(args, "val_root", None),
        "val",
    )
    _, dl, _ = make_loader(
        val_pairs,
        vocab=vocab,
        build_vocab=False,
        batch_size=args.batch_size,
        img_size=img_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    miou = export_predictions(model, dl, args.out_csv, device, compute_iou=True, progress_desc="eval preds")
    if miou is not None:
        print(f"[Eval] mIoU={miou:.4f}")


def predict_loop(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab, img_size = _load_model_from_ckpt(args.ckpt, device)

    test_pairs = resolve_dir_pairs(
        getattr(args, "test_json_dir", None),
        getattr(args, "test_jpg_dir", None),
        getattr(args, "test_root", None),
        "test",
    )
    _, dl, _ = make_loader(
        test_pairs,
        vocab=vocab,
        build_vocab=False,
        batch_size=args.batch_size,
        img_size=img_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    export_predictions(model, dl, args.out_csv, device, compute_iou=False, progress_desc="test preds")

def zip_submission(csv_path: str, zip_path: str):
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        arcname = os.path.basename(csv_path)
        zf.write(csv_path, arcname=arcname)
    print(f"[Submission] Zipped {csv_path} → {zip_path}")


def fit_pipeline(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    test_pairs = resolve_dir_pairs(
        getattr(args, "test_json_dir", None),
        getattr(args, "test_jpg_dir", None),
        getattr(args, "test_root", None),
        "test",
    )

    # ============================================
    # Resume 여부에 따라 model / vocab / loader 구성
    # ============================================
    resume_ckpt = getattr(args, "resume_ckpt", None)

    if resume_ckpt:
        # 기존 체크포인트에서 모델 + vocab + img_size 로드
        model, vocab, used_img_size = _load_model_from_ckpt(resume_ckpt, device)
        ckpt_dim = model.txt.emb.embedding_dim

        # 기존 vocab을 그대로 사용
        train_ds, train_dl, _ = make_loader(
            train_pairs,
            vocab=vocab,
            build_vocab=False,
            batch_size=args.batch_size,
            img_size=used_img_size,
            num_workers=args.num_workers,
            shuffle=True,
        )

        val_ds, val_dl, _ = make_loader(
            val_pairs,
            vocab=vocab,
            build_vocab=False,
            batch_size=args.batch_size,
            img_size=used_img_size,
            num_workers=args.num_workers,
            shuffle=False,
        )
        print(f"[Resume] Loaded checkpoint from {resume_ckpt}")
    else:
        # 처음부터 학습
        used_img_size = args.img_size
        train_ds, train_dl, vocab = make_loader(
            train_pairs,
            vocab=None,
            build_vocab=True,
            batch_size=args.batch_size,
            img_size=used_img_size,
            num_workers=args.num_workers,
            shuffle=True,
        )

        val_ds, val_dl, _ = make_loader(
            val_pairs,
            vocab=vocab,
            build_vocab=False,
            batch_size=args.batch_size,
            img_size=used_img_size,
            num_workers=args.num_workers,
            shuffle=False,
        )

        model = CrossAttnVLM(
            vocab_size=len(vocab.itos),
            dim=args.dim,
            pretrained_backbone=not args.no_pretrain,
            img_size=used_img_size,
        ).to(device)
        ckpt_dim = args.dim

    # NEW: 0.5/1.0 epoch 평가용 eval loader (shuffle=False)
    eval_loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "collate_fn": collate_fn,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": args.num_workers > 0,
    }
    if args.num_workers > 0:
        eval_loader_kwargs["prefetch_factor"] = 8

    val_eval_dl = DataLoader(val_ds, **eval_loader_kwargs)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    # ===== [수정됨] =====
    # best_loss 대신 best_miou 사용 (mIoU는 높을수록 좋으므로 0.0에서 시작)
    best_miou = 0.0
    best_epoch = 0
    patience_ctr = 0

    # NEW: ckpt 이름 베이스
    ckpt_base, ckpt_ext = os.path.splitext(args.save_ckpt)

    # NEW: mIoU eval + ckpt helper
    # val_miou 값을 반환하도록 수정
    def eval_and_ckpt(tag: str, train_iou: Optional[float] = None) -> Optional[float]:
        val_miou = export_predictions(
            model,
            val_eval_dl,
            out_csv="",  # 중간 eval에서는 csv 저장 안 함
            device=device,
            compute_iou=True,
            progress_desc=f"val miou {tag}",
        )
        if val_miou is not None:
            if train_iou is not None:
                print(f"[mIoU {tag}] train={train_iou:.4f}  val={val_miou:.4f}")
            else:
                print(f"[mIoU {tag}] val={val_miou:.4f}")

        ckpt_path = f"{ckpt_base}_{tag}{ckpt_ext}"
        # used_img_size / ckpt_dim 사용
        save_checkpoint(model, vocab, used_img_size, ckpt_path, ckpt_dim, args.no_pretrain)
        return val_miou
    # ===================

    if hasattr(train_dl, "dataset") and hasattr(train_dl.dataset, "__len__"):
        total_samples = len(train_dl.dataset)  # type: ignore[arg-type]
    else:
        total_samples = len(train_dl) * args.batch_size

    for epoch in range(1, args.epochs + 1):
        # 0.5 epoch 지점 콜백: cur_train_iou를 받아서 val mIoU와 함께 찍음
        def half_cb(cur_train_iou, ep=epoch):
            tag = f"ep{ep - 0.5:.1f}"
            eval_and_ckpt(tag, train_iou=cur_train_iou)

        train_loss, train_iou = train_one_epoch(
            model,
            train_dl,
            optimizer,
            scaler,
            device,
            desc=f"train {epoch}/{args.epochs}",
            on_half_epoch=half_cb,  # 여기서 half_cb에 train_iou 전달
        )
        scheduler.step()

        # val_loss는 참고용으로 계산 (Early Stopping 기준 아님)
        val_loss = evaluate_loss(model, val_dl, device, desc="val loss")
        print(f"[Epoch {epoch}/{args.epochs}] train={train_loss:.4f} val={val_loss:.4f}  lr={scheduler.get_last_lr()[0]:.6f}")

        # epoch 끝에서도 한 번 더 전체 val mIoU + ckpt
        full_tag = f"ep{epoch:.1f}"
        val_miou = eval_and_ckpt(full_tag, train_iou=train_iou)

        # ===== [mIoU 기준 Early Stopping 로직] =====
        if val_miou is not None and val_miou >= best_miou - args.early_stop_delta:
            if val_miou > best_miou + args.early_stop_delta:
                patience_ctr = 0
                print(f"[Best mIoU] New best: {val_miou:.4f} (was {best_miou:.4f})")
            else:
                patience_ctr += 1

            best_miou = val_miou
            best_epoch = epoch
            # '같거나' '더 좋을' 때 항상 최신 모델(best)로 저장
            save_checkpoint(model, vocab, used_img_size, args.save_ckpt, ckpt_dim, args.no_pretrain)

        elif val_miou is None:
            print("[Warn] val_miou is None, skipping early stopping check.")
            patience_ctr += 1

        else:
            patience_ctr += 1
            print(f"[EarlyStop] mIoU not improved ({val_miou:.4f} < {best_miou:.4f}). Patience: {patience_ctr}/{args.early_stop_patience}")
            if patience_ctr >= args.early_stop_patience:
                print("[EarlyStop] patience exhausted; halting training")
                break
        # ===== [수정 끝] =====

    # 최종 결과 출력
    print(f"[Best] epoch={best_epoch} val_mIoU={best_miou:.4f}")

    # ===== 이하 부분은 기존 로직 그대로 (최종 저장된 best 모델로 평가/추론) =====
    model, best_vocab, best_img_size = _load_model_from_ckpt(args.save_ckpt, device)

    _, val_eval_dl, _ = make_loader(
        val_pairs,
        vocab=best_vocab,
        build_vocab=False,
        batch_size=args.batch_size,
        img_size=best_img_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    miou = export_predictions(model, val_eval_dl, args.eval_csv, device, compute_iou=True, progress_desc="eval preds")
    if miou is not None:
        print(f"[Eval] mIoU={miou:.4f}")

    _, test_dl, _ = make_loader(
        test_pairs,
        vocab=best_vocab,
        build_vocab=False,
        batch_size=args.batch_size,
        img_size=best_img_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    export_predictions(model, test_dl, args.pred_csv, device, compute_iou=False, progress_desc="test preds")


def get_args():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_common(p):
        p.add_argument("--batch_size", type=int, default=CFG.BATCH_SIZE)
        p.add_argument("--img_size", type=int, default=CFG.IMG_SIZE)
        p.add_argument("--dim", type=int, default=CFG.DIM)
        p.add_argument("--num_workers", type=int, default=CFG.NUM_WORKERS)

    # train
    p_train = sub.add_parser("train")
    add_common(p_train)
    p_train.add_argument("--epochs", type=int, default=CFG.EPOCHS)
    p_train.add_argument("--lr", type=float, default=CFG.LEARNING_RATE)
    p_train.add_argument("--no_pretrain", action="store_true", help="Disable ImageNet pretrained weights")
    p_train.add_argument("--save_ckpt", type=str, default=CFG.CKPT_PATH)
    p_train.add_argument("--train_root", type=str, default=CFG.TRAIN_ROOT,
                         help="High-level directory containing *_json/*_jpg children (auto-discovery)")
    p_train.add_argument("--train_json_dir", type=str, default=CFG.TRAIN_JSON_DIR,
                         help="Explicit training JSON directory (use with --train_jpg_dir)")
    p_train.add_argument("--train_jpg_dir", type=str, default=CFG.TRAIN_JPG_DIR,
                         help="Explicit training JPG directory (use with --train_json_dir)")
    # NEW: resume checkpoint
    p_train.add_argument("--resume_ckpt", type=str, default=None,
                         help="Path to checkpoint to resume training from")

    # eval
    p_eval = sub.add_parser("eval")
    add_common(p_eval)
    p_eval.add_argument("--ckpt", type=str, required=True)
    p_eval.add_argument("--out_csv", type=str, default=CFG.EVAL_CSV)
    p_eval.add_argument("--val_root", type=str, default=CFG.VAL_ROOT,
                        help="High-level validation directory containing *_json/*_jpg children")
    p_eval.add_argument("--val_json_dir", type=str, default=CFG.VAL_JSON_DIR,
                        help="Explicit validation JSON directory (use with --val_jpg_dir)")
    p_eval.add_argument("--val_jpg_dir", type=str, default=CFG.VAL_JPG_DIR,
                        help="Explicit validation JPG directory (use with --val_json_dir)")

    # predict
    p_pred = sub.add_parser("predict")
    add_common(p_pred)
    p_pred.add_argument("--ckpt", type=str, required=True)
    p_pred.add_argument("--out_csv", type=str, default=CFG.PRED_CSV)
    p_pred.add_argument("--test_root", type=str, default=CFG.TEST_ROOT,
                        help="High-level test directory containing *_json/*_jpg children")
    p_pred.add_argument("--test_json_dir", type=str, default=CFG.TEST_JSON_DIR,
                        help="Explicit prediction JSON directory (use with --test_jpg_dir)")
    p_pred.add_argument("--test_jpg_dir", type=str, default=CFG.TEST_JPG_DIR,
                        help="Explicit prediction JPG directory (use with --test_json_dir)")

    # fit (train + eval + predict in one shot)
    p_fit = sub.add_parser("fit")
    add_common(p_fit)
    p_fit.add_argument("--epochs", type=int, default=CFG.EPOCHS)
    p_fit.add_argument("--lr", type=float, default=CFG.LEARNING_RATE)
    p_fit.add_argument("--no_pretrain", action="store_true")
    p_fit.add_argument("--save_ckpt", type=str, default=CFG.CKPT_PATH)
    p_fit.add_argument("--eval_csv", type=str, default=CFG.EVAL_CSV)
    p_fit.add_argument("--pred_csv", type=str, default=CFG.PRED_CSV)
    p_fit.add_argument("--early_stop_patience", type=int, default=5)
    p_fit.add_argument("--early_stop_delta", type=float, default=0.0)
    p_fit.add_argument("--train_root", type=str, default=CFG.TRAIN_ROOT)
    p_fit.add_argument("--train_json_dir", type=str, default=CFG.TRAIN_JSON_DIR)
    p_fit.add_argument("--train_jpg_dir", type=str, default=CFG.TRAIN_JPG_DIR)
    p_fit.add_argument("--val_root", type=str, default=CFG.VAL_ROOT)
    p_fit.add_argument("--val_json_dir", type=str, default=CFG.VAL_JSON_DIR)
    p_fit.add_argument("--val_jpg_dir", type=str, default=CFG.VAL_JPG_DIR)
    p_fit.add_argument("--test_root", type=str, default=CFG.TEST_ROOT)
    p_fit.add_argument("--test_json_dir", type=str, default=CFG.TEST_JSON_DIR)
    p_fit.add_argument("--test_jpg_dir", type=str, default=CFG.TEST_JPG_DIR)
    # NEW: resume checkpoint
    p_fit.add_argument("--resume_ckpt", type=str, default=CFG.RESUME_CKPT_PATH,
                       help="Path to checkpoint to resume training from")

    # submission (zip any csv)
    p_zip = sub.add_parser("zip")
    p_zip.add_argument("--csv", type=str, required=True)
    p_zip.add_argument("--out_zip", type=str, default=CFG.SUBMISSION_ZIP)

    return ap.parse_args()



def main():
    seed_everything(CFG.SEED)
    args = get_args()

    if args.cmd == "train":
        train_loop(args)
    elif args.cmd == "fit":
        fit_pipeline(args)
    elif args.cmd == "eval":
        evaluate_loop(args)
    elif args.cmd == "predict":
        predict_loop(args)
    elif args.cmd == "zip":
        zip_submission(args.csv, args.out_zip)
    else:
        raise ValueError(args.cmd)


if __name__ == "__main__":
    main()