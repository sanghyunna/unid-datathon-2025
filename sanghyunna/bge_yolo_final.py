"""Multimodal bbox regressor powered by DocLayout YOLO + BGE-M3."""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import json
import math
import random
import argparse
import zipfile
from glob import glob
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast

try:
    from transformers import AutoModel, AutoTokenizer

    _HAS_TRANSFORMERS = True
except Exception:
    AutoModel = None
    AutoTokenizer = None
    _HAS_TRANSFORMERS = False

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    _HAS_PEFT = True
except Exception:
    LoraConfig = None
    get_peft_model = None
    prepare_model_for_kbit_training = None
    _HAS_PEFT = False

try:
    from doclayout_yolo import YOLOv10

    _HAS_DOCYOLO = True
except Exception:
    YOLOv10 = None
    _HAS_DOCYOLO = False

try:
    from huggingface_hub import hf_hub_download

    _HAS_HF_HUB = True
except Exception:
    hf_hub_download = None
    _HAS_HF_HUB = False

try:
    from torchvision import transforms as T

    _HAS_TORCHVISION = True
except Exception:
    T = None
    _HAS_TORCHVISION = False


class CFG:
    IMG_SIZE: int = 512
    EPOCHS: int = 17
    LEARNING_RATE: float = 1e-4
    BATCH_SIZE: int = 128
    SEED: int = 42
    NUM_WORKERS: int = 8
    TEXT_ENCODER: str = "BAAI/bge-m3"
    TEXT_MAX_LEN: int = 200
    TEXT_PROMPT: str = ("다음 질문에 답하기 위해서 참조해야할 테이블을 알려주세요. ")
    # TEXT_PROMPT: str = ("다음 질의는 한국어로 작성된 시각적 지시입니다. ")
    CROSS_ATTN_DIM: int = 768
    CROSS_ATTN_LAYERS: int = 1
    CROSS_ATTN_HEADS: int = 4
    LORA_R: int = 16
    LORA_ALPHA: int = 32
    LORA_DROPOUT: float = 0.05
    LORA_TARGET_MODULES: Optional[List[str]] = None  # None = auto-detect
    VISION_ENCODER: str = "doclayout-yolo-small"
    VISION_WEIGHTS: Optional[str] = None
    VISION_REPO_ID: Optional[str] = "juliozhao/DocLayout-YOLO-DocStructBench"
    VISION_FILENAME: Optional[str] = "doclayout_yolo_docstructbench_imgsz1024.pt"
    VISION_CACHE_DIR: Optional[str] = None
    VISION_FEATURE_LEVEL: int = -1
    FREEZE_TEXT: bool = False
    FREEZE_VISION: bool = True
    CKPT_PATH: str = "./outputs/ckpt/layout_regressor_test.pth"
    RESUME_CKPT_PATH: str = "./sanghyunna/outputs/ckpt/_layout_regressor_test_ep3.0.pth"
    # RESUME_CKPT_PATH: str = "./outputs/ckpt/layout_regressor.pth"
    EVAL_CSV: str = "./outputs/preds/eval_pred.csv"
    PRED_CSV: str = "./outputs/preds/test_pred.csv"
    SUBMISSION_ZIP: str = "./outputs/submission.zip"
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
    if not os.path.isfile(path):
        return set()
    bad = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
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
        d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))
    )
    suffixes = [
        "_json",
        "json",
        "_jpg",
        "jpg",
        "_jpeg",
        "jpeg",
        "_img",
        "img",
        "_image",
        "image",
        "_images",
        "images",
        "_query",
        "query",
        "_queries",
        "queries",
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
        if any(tag in lower for tag in ["jpg", "jpeg", "img", "image", "photo"]):
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


def resolve_dir_pairs(
    json_dir: Optional[str],
    jpg_dir: Optional[str],
    root_dir: Optional[str],
    split_name: str,
) -> List[Tuple[str, str]]:
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
        raise ValueError(f"JSON decode error in file: {path}\n{e}") from e


def get_image_path(json_path: str, data: Dict[str, Any], jpg_dir: Optional[str] = None) -> str:
    src = data.get("source_data_info", {})
    jpg_name = src.get("source_data_name_jpg", None)
    if jpg_dir and jpg_name:
        path = os.path.join(jpg_dir, jpg_name)
        if os.path.exists(path):
            return path
    if jpg_name:
        maybe = json_path.replace(os.sep + "json" + os.sep, os.sep + "jpg" + os.sep)
        maybe = os.path.join(os.path.dirname(maybe), jpg_name) if os.path.isdir(os.path.dirname(maybe)) else maybe
        if os.path.exists(maybe):
            return maybe
    base = os.path.splitext(os.path.basename(json_path))[0]
    sibling = os.path.join(os.path.dirname(json_path), base.replace("MI3", "MI2") + ".jpg")
    if os.path.exists(sibling):
        return sibling
    raise FileNotFoundError(f"Could not resolve JPG for {json_path} (jpg_dir={jpg_dir})")


def is_visual_ann(a: Dict[str, Any]) -> bool:
    cid = str(a.get("class_id", "") or "")
    cname = str(a.get("class_name", "") or "")
    has_q = bool(str(a.get("visual_instruction", "") or "").strip())
    looks_visual = cid.startswith("V") or any(k in cname for k in ["표", "차트", "그래프", "chart", "table"])
    return has_q and looks_visual


class UniDSet(Dataset):
    def __init__(
        self,
        json_files: List[str],
        jpg_dir: Optional[str] = None,
        resize_to: Tuple[int, int] = (CFG.IMG_SIZE, CFG.IMG_SIZE),
        supervised_only: bool = False,
    ):
        self.items: List[Dict[str, Any]] = []
        self.resize_to = resize_to
        self.supervised_only = supervised_only
        self.cache_suffix = f"_cache_{resize_to[0]}x{resize_to[1]}.pt"

        desc_name = os.path.basename(jpg_dir) if jpg_dir else (
            os.path.basename(os.path.dirname(json_files[0])) if json_files else "jsons"
        )

        for jf in tqdm(json_files, desc=f"[data] {desc_name}", leave=False, dynamic_ncols=True):
            jf_norm = os.path.normpath(jf)
            if jf_norm in BAD_JSON_WHITELIST:
                print(f"[bad-json] skipping whitelisted JSON: {jf}")
                continue
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
                cname = a.get("class_name", "") or ""
                combined_query = f"{cname} {qtxt}".strip()
                bbox = a.get("bounding_box", None)

                if supervised_only and not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
                    continue

                base, _ = os.path.splitext(img_path)
                cache_path = base + self.cache_suffix

                self.items.append(
                    {
                        "json": jf,
                        "img": img_path,
                        "query_id": qid,
                        "query": combined_query,
                        "bbox": bbox,
                        "class_name": cname,
                        "cache_path": cache_path,
                    }
                )

        if _HAS_TORCHVISION:
            self.tf = T.Compose([T.Resize(resize_to), T.ToTensor()])
        else:
            self.tf = None

    def __len__(self):
        return len(self.items)

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
        img_t: Optional[torch.Tensor] = None
        if cache_path is not None and os.path.isfile(cache_path):
            try:
                cached = torch.load(cache_path, map_location="cpu", weights_only=False)
                if isinstance(cached, dict) and "image" in cached and "orig_size" in cached:
                    img_t = cached["image"]
                    W, H = cached["orig_size"]
                else:
                    raise ValueError("unexpected cache format")
            except Exception as e:
                print(f"[cache] failed to load {cache_path}: {e}")
                img_t = None
        if img_t is None:
            img = Image.open(it["img"]).convert("RGB")
            W, H = img.size
            if self.tf is not None:
                img_t = self.tf(img)
            else:
                img = img.resize(self.resize_to, Image.BILINEAR)
                img_t = self._pil_to_tensor(img)
            if cache_path is not None and not os.path.exists(cache_path):
                try:
                    torch.save({"image": img_t, "orig_size": (W, H)}, cache_path)
                except Exception as e:
                    print(f"[cache] failed to save {cache_path}: {e}")
        sample: Dict[str, Any] = {
            "image": img_t,
            "query_text": it["query"],
            "query_id": it["query_id"],
            "orig_size": (W, H),
            "class_name": it["class_name"],
        }
        bbox = it["bbox"]
        if bbox is not None and isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            x, y, w, h = bbox
            cx = (x + w / 2.0) / W
            cy = (y + h / 2.0) / H
            nw = w / W
            nh = h / H
            target = torch.tensor([cx, cy, nw, nh], dtype=torch.float32)
        else:
            target = None
        sample["target"] = target
        return sample


def prepare_tokenizer(name: str):
    if not _HAS_TRANSFORMERS:
        raise ImportError("transformers is required for the text encoder")
    if AutoTokenizer is None:
        raise ImportError("AutoTokenizer is unavailable despite transformers import")
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.sep_token is not None:
            tokenizer.pad_token = tokenizer.sep_token
        elif tokenizer.cls_token is not None:
            tokenizer.pad_token = tokenizer.cls_token
        else:
            tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = "right"
    return tokenizer


def build_prompt_builder(instruction: Optional[str]) -> Optional[Callable[[str], str]]:
    if instruction is None:
        return None
    prompt = instruction.strip()
    if not prompt:
        return None

    def _builder(text: str) -> str:
        text = text or ""
        return f"{prompt}\n{text}".strip()

    return _builder


def build_collate_fn(
    tokenizer,
    max_text_len: int,
    prompt_builder: Optional[Callable[[str], str]] = None,
):
    def _collate(batch: List[Dict[str, Any]]):
        imgs = torch.stack([b["image"] for b in batch], dim=0)
        texts = [b["query_text"] for b in batch]
        if prompt_builder is not None:
            texts = [prompt_builder(t) for t in texts]
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_text_len,
            return_tensors="pt",
        )
        targets = []
        meta = []
        for b in batch:
            targets.append(b["target"])
            meta.append(
                {
                    "query_id": b["query_id"],
                    "query_text": b["query_text"],
                    "orig_size": b["orig_size"],
                    "class_name": b["class_name"],
                }
            )
        stacked_targets = None
        if all(t is not None for t in targets) and len(targets) > 0:
            stacked_targets = torch.stack(targets, dim=0)
        return imgs, encoded, targets, meta, stacked_targets

    return _collate


class DocLayoutYOLOBackbone(nn.Module):
    def __init__(
        self,
        model_name: str,
        out_dim: int,
        trainable: bool = False,
        feature_level: int = -1,
        weight_path: Optional[str] = None,
        repo_id: Optional[str] = None,
        filename: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        self.project = nn.LazyConv2d(out_dim, 1)
        self.trainable = trainable
        self.feature_level = feature_level
        self.repo_id = repo_id
        self.filename = filename
        self.cache_dir = cache_dir
        self.weights_path: Optional[str] = None
        self._cached_feature: Optional[torch.Tensor] = None
        if not _HAS_DOCYOLO or YOLOv10 is None:
            raise ImportError("doclayout_yolo.YOLOv10 is required for DocLayoutYOLOBackbone")

        weight_file = weight_path
        if weight_file is None:
            if repo_id is None or filename is None:
                raise ValueError(
                    "DocLayout YOLO backbone requires either --vision_weights or both --vision_repo_id/--vision_filename"
                )
            if not _HAS_HF_HUB or hf_hub_download is None:
                raise ImportError("huggingface_hub is required to download DocLayout YOLO weights")
            weight_file = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=cache_dir,
                resume_download=True,
            )
        if not os.path.isfile(weight_file):
            raise FileNotFoundError(f"DocLayout YOLO weight not found: {weight_file}")
        self.weights_path = weight_file

        try:
            yolo_wrapper = YOLOv10(weight_file)
        except Exception as exc:
            raise RuntimeError(f"DocLayout YOLO failed to load weights {weight_file}") from exc
        self.detector_model = getattr(yolo_wrapper, "model", None)
        if self.detector_model is None or not hasattr(self.detector_model, "model"):
            raise AttributeError("YOLOv10 wrapper missing nested 'model' attribute")
        if not trainable:
            self.detector_model.eval()
            for p in self.detector_model.parameters():
                p.requires_grad = False
        detect_module = self.detector_model.model[-1]
        detect_module.register_forward_pre_hook(self._capture_hook)

    def _capture_hook(self, module, inputs):
        feats = inputs[0]
        if isinstance(feats, (list, tuple)) and feats:
            idx = self.feature_level
            if idx < 0:
                idx = len(feats) + idx
            idx = max(0, min(idx, len(feats) - 1))
            candidate = feats[idx]
        else:
            candidate = feats
        if not isinstance(candidate, torch.Tensor):
            raise TypeError("DocLayout YOLO hook expected tensor features")
        self._cached_feature = candidate

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.detector_model is None:
            raise RuntimeError("DocLayout YOLO backbone is not initialized")
        self.detector_model.to(images.device)
        with torch.set_grad_enabled(self.trainable):
            _ = self.detector_model(images)
        if self._cached_feature is None:
            raise RuntimeError("DocLayout YOLO forward hook did not capture feature map")
        feat = self._cached_feature
        self._cached_feature = None
        return self.project(feat)


class BGEM3TextEncoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        dim: int,
        dropout: float = 0.1,
        freeze: bool = False,
        use_lora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None,
    ):
        super().__init__()
        if not _HAS_TRANSFORMERS:
            raise ImportError("transformers is required for BGEM3TextEncoder")
        if AutoModel is None:
            raise ImportError("AutoModel is unavailable despite transformers import")
        
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = getattr(self.encoder.config, "hidden_size", dim)
        self.proj = nn.Linear(hidden, dim)
        self.dropout = nn.Dropout(dropout)
        self.model_name = model_name
        self.use_lora = use_lora
        
        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False
        elif use_lora:
            # Apply QLoRA when not frozen
            if not _HAS_PEFT:
                raise ImportError("peft is required for LoRA. Install with: pip install peft")
            if LoraConfig is None or get_peft_model is None:
                raise ImportError("peft components unavailable")
            
            # Freeze base model first
            for p in self.encoder.parameters():
                p.requires_grad = False
            
            # Configure LoRA
            if lora_target_modules is None:
                # Auto-detect target modules for BERT-like models
                lora_target_modules = ["query", "key", "value"]
            
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="FEATURE_EXTRACTION",
            )
            
            self.encoder = get_peft_model(self.encoder, lora_config)
            print(f"[LoRA] Applied to text encoder with r={lora_r}, alpha={lora_alpha}")
            self.encoder.print_trainable_parameters()

    def forward(self, tokens: Dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = self.encoder(**tokens)
        if getattr(outputs, "pooler_output", None) is not None:
            pooled = outputs.pooler_output
        else:
            pooled = outputs.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)
        return self.proj(pooled)


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.k_proj = nn.Conv2d(dim, dim, 1)
        self.v_proj = nn.Conv2d(dim, dim, 1)
        self.q_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, q_vec: torch.Tensor, fmap: torch.Tensor) -> torch.Tensor:
        B, D, H, W = fmap.shape
        k = self.k_proj(fmap).flatten(2).transpose(1, 2)
        v = self.v_proj(fmap).flatten(2).transpose(1, 2)
        q = self.q_norm(q_vec).unsqueeze(1)
        attn_out, _ = self.attn(q, k, v)
        attn_out = attn_out.squeeze(1)
        q_vec = q_vec + self.dropout(attn_out)
        ffn_out = self.ffn(q_vec)
        q_vec = q_vec + self.dropout(ffn_out)
        return q_vec


class CrossAttentionRegressor(nn.Module):
    def __init__(
        self,
        dim: int,
        num_blocks: int,
        heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([CrossAttentionBlock(dim, heads, dropout) for _ in range(num_blocks)])
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, 4),
        )

    def forward(self, q_vec: torch.Tensor, fmap: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            q_vec = block(q_vec, fmap)
        pred = self.head(q_vec)
        return torch.sigmoid(pred)


class LayoutAwareRegressor(nn.Module):
    def __init__(
        self,
        cross_attn_dim: int,
        text_encoder_name: str,
        vision_encoder_name: str,
        vision_weights: Optional[str],
        vision_repo_id: Optional[str],
        vision_filename: Optional[str],
        vision_cache_dir: Optional[str],
        cross_attn_layers: int,
        cross_attn_heads: int,
        freeze_text: bool,
        freeze_vision: bool,
        vision_feature_level: int,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None,
    ):
        super().__init__()
        # When freeze_text=False, apply LoRA
        use_lora = not freeze_text
        self.text = BGEM3TextEncoder(
            text_encoder_name, 
            dim=cross_attn_dim, 
            freeze=freeze_text,
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_modules=lora_target_modules,
        )
        self.vision = DocLayoutYOLOBackbone(
            vision_encoder_name,
            out_dim=cross_attn_dim,
            trainable=not freeze_vision,
            feature_level=vision_feature_level,
            weight_path=vision_weights,
            repo_id=vision_repo_id,
            filename=vision_filename,
            cache_dir=vision_cache_dir,
        )
        self.head = CrossAttentionRegressor(cross_attn_dim, cross_attn_layers, heads=cross_attn_heads)
        self.dim = cross_attn_dim
        self.text_name = text_encoder_name
        self.vision_name = vision_encoder_name
        self.vision_weights_path = getattr(self.vision, "weights_path", vision_weights)
        self.vision_repo_id = vision_repo_id
        self.vision_filename = vision_filename
        self.vision_cache_dir = vision_cache_dir
        self.cross_attn_layers = cross_attn_layers
        self.num_cross_attn = cross_attn_layers  # backward compat for checkpoints
        self.cross_attn_heads = cross_attn_heads
        self.freeze_text = freeze_text
        self.freeze_vision = freeze_vision
        self.vision_feature_level = vision_feature_level

    def forward(self, images: torch.Tensor, tokens: Dict[str, torch.Tensor]) -> torch.Tensor:
        text_tokens = {k: v for k, v in tokens.items()}
        q = self.text(text_tokens)
        fmap = self.vision(images)
        pred_norm = self.head(q, fmap)
        return pred_norm


def batch_iou_norm(pred: torch.Tensor, tgt: torch.Tensor) -> float:
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


def export_predictions(
    model: nn.Module,
    loader: DataLoader,
    out_csv: str,
    device: torch.device,
    compute_iou: bool = False,
    progress_desc: str = "predict",
) -> Optional[float]:
    rows = []
    ious: List[float] = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        iterator = tqdm(loader, desc=progress_desc, leave=False, dynamic_ncols=True)
        for imgs, text_inputs, targets, meta, _ in iterator:
            imgs = imgs.to(device, non_blocking=True)
            text_inputs = {k: v.to(device, non_blocking=True) for k, v in text_inputs.items()}
            pred = model(imgs, text_inputs)
            for i in range(imgs.size(0)):
                W, H = meta[i]["orig_size"]
                cx, cy, nw, nh = [float(v) for v in pred[i].cpu().numpy().tolist()]
                x = (cx - nw / 2.0) * W
                y = (cy - nh / 2.0) * H
                w = nw * W
                h = nh * H
                rows.append(
                    {
                        "query_id": meta[i]["query_id"],
                        "query_text": meta[i]["query_text"],
                        "pred_x": x,
                        "pred_y": y,
                        "pred_w": w,
                        "pred_h": h,
                    }
                )
                if compute_iou and targets[i] is not None:
                    gt = [float(v) for v in targets[i].numpy().tolist()]
                    gx = (gt[0] - gt[2] / 2.0) * W
                    gy = (gt[1] - gt[3] / 2.0) * H
                    gw = gt[2] * W
                    gh = gt[3] * H
                    px = x
                    py = y
                    pw = w
                    ph = h
                    px2 = px + pw
                    py2 = py + ph
                    gx2 = gx + gw
                    gy2 = gy + gh
                    ix1 = max(px, gx)
                    iy1 = max(py, gy)
                    ix2 = min(px2, gx2)
                    iy2 = min(py2, gy2)
                    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                    union = pw * ph + gw * gh - inter if (pw * ph + gw * gh - inter) > 0 else 1e-6
                    ious.append(inter / union)
    if out_csv:
        out_dir = os.path.dirname(out_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        import pandas as pd

        df = pd.DataFrame(
            rows,
            columns=["query_id", "query_text", "pred_x", "pred_y", "pred_w", "pred_h"],
        )
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"[Saved] {out_csv}")
    miou: Optional[float] = None
    if compute_iou and ious:
        miou = float(np.mean(ious))
    elif compute_iou:
        print("[Eval] No ground-truth boxes available; IoU skipped.")
    if was_training:
        model.train()
    return miou


def make_loader(
    dir_pairs: List[Tuple[str, str]],
    tokenizer,
    prompt_builder: Optional[Callable[[str], str]],
    batch_size: int,
    img_size: int,
    max_text_len: int,
    num_workers: int,
    shuffle: bool,
    supervised_only: bool,
):
    if not dir_pairs:
        raise ValueError("dir_pairs must contain at least one (json_dir, jpg_dir) tuple")
    datasets: List[Dataset] = []
    for json_dir, jpg_dir in dir_pairs:
        json_files = find_jsons(json_dir)
        subset = UniDSet(
            json_files,
            jpg_dir=jpg_dir,
            resize_to=(img_size, img_size),
            supervised_only=supervised_only,
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
    collate_fn = build_collate_fn(tokenizer, max_text_len, prompt_builder)
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "collate_fn": collate_fn,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    dl = DataLoader(full_ds, **loader_kwargs)
    return full_ds, dl


def save_checkpoint(
    model: nn.Module,
    tokenizer_name: str,
    img_size: int,
    ckpt_path: str,
    dim: int,
    text_max_len: int,
    text_prompt: Optional[str],
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    scaler: Optional[GradScaler] = None,
    epoch: Optional[float] = None,
    best_miou: Optional[float] = None,
    patience_counter: Optional[int] = None,
):
    os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
    cross_dim = getattr(model, "dim", dim)
    cross_layers = getattr(model, "cross_attn_layers", getattr(model, "num_cross_attn", CFG.CROSS_ATTN_LAYERS))
    cross_heads = getattr(model, "cross_attn_heads", CFG.CROSS_ATTN_HEADS)
    
    # Save LoRA parameters from text encoder if available
    lora_r = CFG.LORA_R
    lora_alpha = CFG.LORA_ALPHA
    lora_dropout = CFG.LORA_DROPOUT
    lora_target_modules = CFG.LORA_TARGET_MODULES
    if hasattr(model, "text") and hasattr(model.text, "use_lora") and model.text.use_lora:
        if hasattr(model.text.encoder, "peft_config"):
            peft_cfg = list(model.text.encoder.peft_config.values())[0]
            lora_r = peft_cfg.r
            lora_alpha = peft_cfg.lora_alpha
            lora_dropout = peft_cfg.lora_dropout
            lora_target_modules = peft_cfg.target_modules
    
    ckpt = {
        "model_state": model.state_dict(),
        "tokenizer_name": tokenizer_name,
        "img_size": img_size,
        "dim": cross_dim,
        "cross_attn_dim": cross_dim,
        "text_max_len": text_max_len,
        "text_prompt": text_prompt,
        "text_encoder_name": getattr(model, "text_name", CFG.TEXT_ENCODER),
        "vision_encoder_name": getattr(model, "vision_name", CFG.VISION_ENCODER),
        "num_cross_attn": cross_layers,
        "cross_attn_layers": cross_layers,
        "cross_attn_heads": cross_heads,
        "freeze_text": getattr(model, "freeze_text", CFG.FREEZE_TEXT),
        "freeze_vision": getattr(model, "freeze_vision", CFG.FREEZE_VISION),
        "vision_feature_level": getattr(model, "vision_feature_level", CFG.VISION_FEATURE_LEVEL),
        "vision_weights_path": getattr(model, "vision_weights_path", CFG.VISION_WEIGHTS),
        "vision_repo_id": getattr(model, "vision_repo_id", CFG.VISION_REPO_ID),
        "vision_filename": getattr(model, "vision_filename", CFG.VISION_FILENAME),
        "vision_cache_dir": getattr(model, "vision_cache_dir", CFG.VISION_CACHE_DIR),
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "lora_target_modules": lora_target_modules,
    }
    if optimizer is not None:
        ckpt["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        ckpt["scheduler_state"] = scheduler.state_dict()
    if scaler is not None:
        ckpt["scaler_state"] = scaler.state_dict()
    if epoch is not None:
        ckpt["epoch"] = float(epoch)
    if best_miou is not None:
        ckpt["best_miou"] = float(best_miou)
    if patience_counter is not None:
        ckpt["patience_counter"] = int(patience_counter)
    torch.save(ckpt, ckpt_path)
    print(f"[Saved] {ckpt_path}")


def _load_model_from_ckpt(
    ckpt_path: str,
    device: torch.device,
    freeze_text: Optional[bool] = None,
    freeze_vision: Optional[bool] = None,
):
    ckpt = torch.load(ckpt_path, map_location=device)
    text_name = ckpt.get("text_encoder_name", CFG.TEXT_ENCODER)
    vision_name = ckpt.get("vision_encoder_name", CFG.VISION_ENCODER)
    cross_dim = ckpt.get("cross_attn_dim", ckpt.get("dim", CFG.CROSS_ATTN_DIM))
    cross_layers = ckpt.get("cross_attn_layers", ckpt.get("num_cross_attn", CFG.CROSS_ATTN_LAYERS))
    cross_heads = ckpt.get("cross_attn_heads", CFG.CROSS_ATTN_HEADS)
    vision_level = ckpt.get("vision_feature_level", CFG.VISION_FEATURE_LEVEL)
    vision_weights = ckpt.get("vision_weights_path", ckpt.get("vision_weights", CFG.VISION_WEIGHTS))
    vision_repo_id = ckpt.get("vision_repo_id", CFG.VISION_REPO_ID)
    vision_filename = ckpt.get("vision_filename", CFG.VISION_FILENAME)
    vision_cache_dir = ckpt.get("vision_cache_dir", CFG.VISION_CACHE_DIR)
    freeze_text = freeze_text if freeze_text is not None else ckpt.get("freeze_text", CFG.FREEZE_TEXT)
    freeze_vision = (
        freeze_vision if freeze_vision is not None else ckpt.get("freeze_vision", CFG.FREEZE_VISION)
    )
    freeze_text = bool(freeze_text)
    freeze_vision = bool(freeze_vision)
    
    lora_r = ckpt.get("lora_r", CFG.LORA_R)
    lora_alpha = ckpt.get("lora_alpha", CFG.LORA_ALPHA)
    lora_dropout = ckpt.get("lora_dropout", CFG.LORA_DROPOUT)
    lora_target_modules = ckpt.get("lora_target_modules", CFG.LORA_TARGET_MODULES)
    
    model = LayoutAwareRegressor(
        cross_attn_dim=cross_dim,
        text_encoder_name=text_name,
        vision_encoder_name=vision_name,
        vision_weights=vision_weights,
        vision_repo_id=vision_repo_id,
        vision_filename=vision_filename,
        vision_cache_dir=vision_cache_dir,
        cross_attn_layers=cross_layers,
        cross_attn_heads=cross_heads,
        freeze_text=freeze_text,
        freeze_vision=freeze_vision,
        vision_feature_level=vision_level,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    meta = {
        "tokenizer_name": ckpt.get("tokenizer_name", text_name),
        "img_size": ckpt.get("img_size", CFG.IMG_SIZE),
        "dim": cross_dim,
        "cross_attn_dim": cross_dim,
        "text_max_len": ckpt.get("text_max_len", CFG.TEXT_MAX_LEN),
        "text_prompt": ckpt.get("text_prompt", None),
        "text_encoder_name": text_name,
        "vision_encoder_name": vision_name,
        "num_cross_attn": cross_layers,
        "cross_attn_layers": cross_layers,
        "cross_attn_heads": cross_heads,
        "freeze_text": freeze_text,
        "freeze_vision": freeze_vision,
        "vision_feature_level": vision_level,
        "vision_weights_path": vision_weights,
        "vision_repo_id": vision_repo_id,
        "vision_filename": vision_filename,
        "vision_cache_dir": vision_cache_dir,
    }
    train_state = {
        "optimizer_state": ckpt.get("optimizer_state"),
        "scheduler_state": ckpt.get("scheduler_state"),
        "scaler_state": ckpt.get("scaler_state"),
        "epoch": ckpt.get("epoch"),
        "best_miou": ckpt.get("best_miou"),
        "patience_counter": ckpt.get("patience_counter"),
    }
    return model, meta, train_state


def infer_start_epoch(train_state: Optional[Dict[str, Any]], default_start: int = 1) -> int:
    state = train_state or {}
    marker = state.get("epoch")
    if marker is None:
        return default_start
    try:
        marker = float(marker)
    except (TypeError, ValueError):
        return default_start
    return max(default_start, int(math.floor(marker)) + 1)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    desc: Optional[str] = None,
    on_half_epoch: Optional[Callable[[float], None]] = None,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_iou = 0.0
    total = 0
    num_batches = len(loader)
    half_point = num_batches // 2
    iterator = tqdm(loader, desc=desc or "train", leave=False, dynamic_ncols=True)
    for step, (imgs, text_inputs, targets, _, stacked_targets) in enumerate(iterator, start=1):
        imgs = imgs.to(device, non_blocking=True)
        text_inputs = {k: v.to(device, non_blocking=True) for k, v in text_inputs.items()}
        if stacked_targets is not None:
            t = stacked_targets.to(device, non_blocking=True)
        else:
            filtered = [tar for tar in targets if tar is not None]
            if not filtered:
                continue
            t = torch.stack(filtered, dim=0).to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        use_autocast = imgs.is_cuda and torch.cuda.is_available()
        amp_dtype = torch.bfloat16
        if use_autocast and not torch.cuda.is_bf16_supported():
            amp_dtype = torch.float16
        with autocast(enabled=use_autocast, dtype=amp_dtype):
            pred = model(imgs, text_inputs)
            if pred.shape[0] != t.shape[0]:
                pred = pred[: t.shape[0]]
            loss = F.smooth_l1_loss(pred, t, reduction="mean")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        B = t.size(0)
        running_loss += float(loss.item()) * B
        total += B
        with torch.no_grad():
            batch_iou = batch_iou_norm(pred.detach(), t)
        running_iou += batch_iou * B
        if on_half_epoch is not None and half_point > 0 and step == half_point:
            cur_train_iou = running_iou / max(1, total)
            on_half_epoch(cur_train_iou)
    avg_loss = running_loss / max(1, total)
    avg_iou = running_iou / max(1, total)
    return avg_loss, avg_iou


def train_loop(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resume_ckpt = getattr(args, "resume_ckpt", None)
    train_state: Dict[str, Any] = {}
    if resume_ckpt:
        model, meta, train_state = _load_model_from_ckpt(resume_ckpt, device)
        tokenizer_name = meta["tokenizer_name"]
        used_img_size = meta["img_size"]
        text_max_len = meta["text_max_len"]
        prompt_text = meta.get("text_prompt")
        prompt_builder = build_prompt_builder(prompt_text)
        tokenizer = prepare_tokenizer(tokenizer_name)
        print(f"[Resume] Loaded checkpoint from {resume_ckpt}")
    else:
        tokenizer_name = args.text_encoder_name
        used_img_size = args.img_size
        text_max_len = args.text_max_len
        prompt_text = args.text_prompt
        prompt_builder = build_prompt_builder(prompt_text)
        tokenizer = prepare_tokenizer(tokenizer_name)
        model = LayoutAwareRegressor(
            cross_attn_dim=args.cross_attn_dim,
            text_encoder_name=args.text_encoder_name,
            vision_encoder_name=args.vision_encoder_name,
            vision_weights=args.vision_weights,
            vision_repo_id=args.vision_repo_id,
            vision_filename=args.vision_filename,
            vision_cache_dir=args.vision_cache_dir,
            cross_attn_layers=args.cross_attn_layers,
            cross_attn_heads=args.cross_attn_heads,
            freeze_text=args.freeze_text,
            freeze_vision=args.freeze_vision,
            vision_feature_level=args.vision_feature_level,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lora_target_modules=args.lora_target_modules,
        ).to(device)
    train_pairs = resolve_dir_pairs(
        getattr(args, "train_json_dir", None),
        getattr(args, "train_jpg_dir", None),
        getattr(args, "train_root", None),
        "train",
    )
    train_ds, train_dl = make_loader(
        train_pairs,
        tokenizer,
        prompt_builder,
        batch_size=args.batch_size,
        img_size=used_img_size,
        max_text_len=text_max_len,
        num_workers=args.num_workers,
        shuffle=True,
        supervised_only=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(enabled=torch.cuda.is_available())
    if train_state.get("optimizer_state"):
        optimizer.load_state_dict(train_state["optimizer_state"])
    if train_state.get("scheduler_state"):
        scheduler.load_state_dict(train_state["scheduler_state"])
    if train_state.get("scaler_state"):
        scaler.load_state_dict(train_state["scaler_state"])
    start_epoch = infer_start_epoch(train_state, 1)
    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_iou = train_one_epoch(
            model,
            train_dl,
            optimizer,
            scaler,
            device,
            desc=f"train epoch {epoch}/{args.epochs}",
        )
        scheduler.step()
        last_epoch = epoch
        avg = train_loss
        print(f"[Epoch {epoch}/{args.epochs}] loss={avg:.4f}  lr={scheduler.get_last_lr()[0]:.6f}  miou={train_iou:.4f}")
    save_checkpoint(
        model,
        tokenizer_name,
        used_img_size,
        args.save_ckpt,
        dim=model.dim,
        text_max_len=text_max_len,
        text_prompt=prompt_text,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=last_epoch,
    )


def evaluate_loop(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta, _ = _load_model_from_ckpt(args.ckpt, device)
    tokenizer = prepare_tokenizer(meta["tokenizer_name"])
    prompt_builder = build_prompt_builder(meta.get("text_prompt"))
    val_pairs = resolve_dir_pairs(
        getattr(args, "val_json_dir", None),
        getattr(args, "val_jpg_dir", None),
        getattr(args, "val_root", None),
        "val",
    )
    _, dl = make_loader(
        val_pairs,
        tokenizer,
        prompt_builder,
        batch_size=args.batch_size,
        img_size=meta["img_size"],
        max_text_len=meta["text_max_len"],
        num_workers=args.num_workers,
        shuffle=False,
        supervised_only=False,
    )
    miou = export_predictions(model, dl, args.out_csv, device, compute_iou=True, progress_desc="eval preds")
    if miou is not None:
        print(f"[Eval] mIoU={miou:.4f}")


def predict_loop(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta, _ = _load_model_from_ckpt(args.ckpt, device)
    tokenizer = prepare_tokenizer(meta["tokenizer_name"])
    prompt_builder = build_prompt_builder(meta.get("text_prompt"))
    test_pairs = resolve_dir_pairs(
        getattr(args, "test_json_dir", None),
        getattr(args, "test_jpg_dir", None),
        getattr(args, "test_root", None),
        "test",
    )
    _, dl = make_loader(
        test_pairs,
        tokenizer,
        prompt_builder,
        batch_size=args.batch_size,
        img_size=meta["img_size"],
        max_text_len=meta["text_max_len"],
        num_workers=args.num_workers,
        shuffle=False,
        supervised_only=False,
    )
    export_predictions(model, dl, args.out_csv, device, compute_iou=False, progress_desc="test preds")


def zip_submission(csv_path: str, zip_path: str):
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        arcname = os.path.basename(csv_path)
        zf.write(csv_path, arcname=arcname)
    print(f"[Submission] Zipped {csv_path} → {zip_path}")


def fit_pipeline(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resume_ckpt = getattr(args, "resume_ckpt", None)
    train_state: Dict[str, Any] = {}
    if resume_ckpt:
        model, meta, train_state = _load_model_from_ckpt(resume_ckpt, device)
        tokenizer_name = meta["tokenizer_name"]
        used_img_size = meta["img_size"]
        text_max_len = meta["text_max_len"]
        prompt_text = meta.get("text_prompt")
        prompt_builder = build_prompt_builder(prompt_text)
        tokenizer = prepare_tokenizer(tokenizer_name)
        print(f"[Resume] Loaded checkpoint from {resume_ckpt}")
    else:
        tokenizer_name = args.text_encoder_name
        used_img_size = args.img_size
        text_max_len = args.text_max_len
        prompt_text = args.text_prompt
        prompt_builder = build_prompt_builder(prompt_text)
        tokenizer = prepare_tokenizer(tokenizer_name)
        model = LayoutAwareRegressor(
            cross_attn_dim=args.cross_attn_dim,
            text_encoder_name=args.text_encoder_name,
            vision_encoder_name=args.vision_encoder_name,
            vision_weights=args.vision_weights,
            vision_repo_id=args.vision_repo_id,
            vision_filename=args.vision_filename,
            vision_cache_dir=args.vision_cache_dir,
            cross_attn_layers=args.cross_attn_layers,
            cross_attn_heads=args.cross_attn_heads,
            freeze_text=args.freeze_text,
            freeze_vision=args.freeze_vision,
            vision_feature_level=args.vision_feature_level,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lora_target_modules=args.lora_target_modules,
        ).to(device)
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
    train_ds, train_dl = make_loader(
        train_pairs,
        tokenizer,
        prompt_builder,
        batch_size=args.batch_size,
        img_size=used_img_size,
        max_text_len=text_max_len,
        num_workers=args.num_workers,
        shuffle=True,
        supervised_only=True,
    )
    _, val_dl = make_loader(
        val_pairs,
        tokenizer,
        prompt_builder,
        batch_size=args.batch_size,
        img_size=used_img_size,
        max_text_len=text_max_len,
        num_workers=args.num_workers,
        shuffle=False,
        supervised_only=False,
    )
    _, test_dl = make_loader(
        test_pairs,
        tokenizer,
        prompt_builder,
        batch_size=args.batch_size,
        img_size=used_img_size,
        max_text_len=text_max_len,
        num_workers=args.num_workers,
        shuffle=False,
        supervised_only=False,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(enabled=torch.cuda.is_available())
    if train_state.get("optimizer_state"):
        optimizer.load_state_dict(train_state["optimizer_state"])
    if train_state.get("scheduler_state"):
        scheduler.load_state_dict(train_state["scheduler_state"])
    if train_state.get("scaler_state"):
        scaler.load_state_dict(train_state["scaler_state"])
    best_miou = float(train_state.get("best_miou") or 0.0)
    patience_ctr = int(train_state.get("patience_counter") or 0)
    ckpt_base, ckpt_ext = os.path.splitext(args.save_ckpt)
    pred_csv_base, pred_csv_ext = os.path.splitext(args.pred_csv)
    start_epoch = infer_start_epoch(train_state, 1)

    def eval_and_ckpt(tag: str, train_iou: Optional[float] = None) -> Tuple[Optional[float], str]:
        ckpt_path = f"{ckpt_base}_{tag}{ckpt_ext}"
        miou = export_predictions(
            model,
            val_dl,
            out_csv="",
            device=device,
            compute_iou=True,
            progress_desc=f"val miou {tag}",
        )
        if miou is not None:
            if train_iou is not None:
                print(f"[mIoU {tag}] train={train_iou:.4f}  val={miou:.4f}")
            else:
                print(f"[mIoU {tag}] val={miou:.4f}")
        return miou, ckpt_path
    for epoch in range(start_epoch, args.epochs + 1):
        def half_cb(cur_train_iou, ep=epoch):
            tag = f"ep{ep - 0.5:.1f}"
            _, ckpt_path = eval_and_ckpt(tag, train_iou=cur_train_iou)
            save_checkpoint(
                model,
                tokenizer_name,
                used_img_size,
                ckpt_path,
                dim=model.dim,
                text_max_len=text_max_len,
                text_prompt=prompt_text,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=ep - 0.5,
                best_miou=best_miou,
                patience_counter=patience_ctr,
            )

        train_loss, train_iou = train_one_epoch(
            model,
            train_dl,
            optimizer,
            scaler,
            device,
            desc=f"train {epoch}/{args.epochs}",
            on_half_epoch=half_cb,
        )
        scheduler.step()
        print(
            f"[Epoch {epoch}/{args.epochs}] train={train_loss:.4f} "
            f"train_mIoU={train_iou:.4f}  lr={scheduler.get_last_lr()[0]:.6f}"
        )
        full_tag = f"ep{epoch:.1f}"
        val_miou, tag_ckpt_path = eval_and_ckpt(full_tag, train_iou=train_iou)
        
        # Create test predictions after each epoch
        test_pred_csv = f"{pred_csv_base}_ep{epoch}{pred_csv_ext}"
        export_predictions(
            model,
            test_dl,
            test_pred_csv,
            device,
            compute_iou=False,
            progress_desc=f"test preds ep{epoch}"
        )
        
        should_stop = False
        if val_miou is not None and val_miou >= best_miou - args.early_stop_delta:
            if val_miou > best_miou + args.early_stop_delta:
                patience_ctr = 0
                print(f"[Best mIoU] New best: {val_miou:.4f} (was {best_miou:.4f})")
            else:
                patience_ctr += 1
            best_miou = max(best_miou, val_miou)
            save_checkpoint(
                model,
                tokenizer_name,
                used_img_size,
                args.save_ckpt,
                dim=model.dim,
                text_max_len=text_max_len,
                text_prompt=prompt_text,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_miou=best_miou,
                patience_counter=patience_ctr,
            )
        elif val_miou is None:
            print("[Warn] val_miou is None, skipping early stopping check.")
            patience_ctr += 1
        else:
            patience_ctr += 1
            print(
                f"[EarlyStop] mIoU not improved ({val_miou:.4f} < {best_miou:.4f}). "
                f"Patience: {patience_ctr}/{args.early_stop_patience}"
            )
            if patience_ctr >= args.early_stop_patience:
                print("[EarlyStop] patience exhausted; halting training")
                should_stop = True
        save_checkpoint(
            model,
            tokenizer_name,
            used_img_size,
            tag_ckpt_path,
            dim=model.dim,
            text_max_len=text_max_len,
            text_prompt=prompt_text,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_miou=best_miou,
            patience_counter=patience_ctr,
        )
        if should_stop:
            break
    print(f"[Best] val_mIoU={best_miou:.4f}")
    model, meta, _ = _load_model_from_ckpt(args.save_ckpt, device)
    tokenizer = prepare_tokenizer(meta["tokenizer_name"])
    prompt_builder = build_prompt_builder(meta.get("text_prompt"))
    _, val_eval_dl = make_loader(
        val_pairs,
        tokenizer,
        prompt_builder,
        batch_size=args.batch_size,
        img_size=meta["img_size"],
        max_text_len=meta["text_max_len"],
        num_workers=args.num_workers,
        shuffle=False,
        supervised_only=False,
    )
    miou = export_predictions(model, val_eval_dl, args.eval_csv, device, compute_iou=True, progress_desc="eval preds")
    if miou is not None:
        print(f"[Eval] mIoU={miou:.4f}")
    _, test_dl = make_loader(
        test_pairs,
        tokenizer,
        prompt_builder,
        batch_size=args.batch_size,
        img_size=meta["img_size"],
        max_text_len=meta["text_max_len"],
        num_workers=args.num_workers,
        shuffle=False,
        supervised_only=False,
    )
    export_predictions(model, test_dl, args.pred_csv, device, compute_iou=False, progress_desc="test preds")


def get_args():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_common(p):
        p.add_argument("--batch_size", type=int, default=CFG.BATCH_SIZE)
        p.add_argument("--img_size", type=int, default=CFG.IMG_SIZE)
        p.add_argument("--cross_attn_dim", type=int, default=CFG.CROSS_ATTN_DIM)
        p.add_argument("--num_workers", type=int, default=CFG.NUM_WORKERS)
        p.add_argument("--text_encoder_name", type=str, default=CFG.TEXT_ENCODER)
        p.add_argument("--vision_encoder_name", type=str, default=CFG.VISION_ENCODER)
        p.add_argument("--vision_weights", type=str, default=CFG.VISION_WEIGHTS)
        p.add_argument("--vision_repo_id", type=str, default=CFG.VISION_REPO_ID)
        p.add_argument("--vision_filename", type=str, default=CFG.VISION_FILENAME)
        p.add_argument("--vision_cache_dir", type=str, default=CFG.VISION_CACHE_DIR)
        p.add_argument("--text_max_len", type=int, default=CFG.TEXT_MAX_LEN)
        p.add_argument("--text_prompt", type=str, default=CFG.TEXT_PROMPT)
        p.add_argument("--cross_attn_layers", type=int, default=CFG.CROSS_ATTN_LAYERS)
        p.add_argument("--cross_attn_heads", type=int, default=CFG.CROSS_ATTN_HEADS)
        p.add_argument("--vision_feature_level", type=int, default=CFG.VISION_FEATURE_LEVEL)
        p.add_argument("--freeze_text", action="store_true", default=CFG.FREEZE_TEXT)
        p.add_argument("--freeze_vision", action="store_true", default=CFG.FREEZE_VISION)
        p.add_argument("--lora_r", type=int, default=CFG.LORA_R)
        p.add_argument("--lora_alpha", type=int, default=CFG.LORA_ALPHA)
        p.add_argument("--lora_dropout", type=float, default=CFG.LORA_DROPOUT)
        p.add_argument("--lora_target_modules", type=str, nargs="*", default=CFG.LORA_TARGET_MODULES)

    p_train = sub.add_parser("train")
    add_common(p_train)
    p_train.add_argument("--epochs", type=int, default=CFG.EPOCHS)
    p_train.add_argument("--lr", type=float, default=CFG.LEARNING_RATE)
    p_train.add_argument("--save_ckpt", type=str, default=CFG.CKPT_PATH)
    p_train.add_argument("--train_root", type=str, default=CFG.TRAIN_ROOT)
    p_train.add_argument("--train_json_dir", type=str, default=CFG.TRAIN_JSON_DIR)
    p_train.add_argument("--train_jpg_dir", type=str, default=CFG.TRAIN_JPG_DIR)
    p_train.add_argument("--resume_ckpt", type=str, default=None)

    p_eval = sub.add_parser("eval")
    add_common(p_eval)
    p_eval.add_argument("--ckpt", type=str, required=True)
    p_eval.add_argument("--out_csv", type=str, default=CFG.EVAL_CSV)
    p_eval.add_argument("--val_root", type=str, default=CFG.VAL_ROOT)
    p_eval.add_argument("--val_json_dir", type=str, default=CFG.VAL_JSON_DIR)
    p_eval.add_argument("--val_jpg_dir", type=str, default=CFG.VAL_JPG_DIR)

    p_pred = sub.add_parser("predict")
    add_common(p_pred)
    p_pred.add_argument("--ckpt", type=str, required=True)
    p_pred.add_argument("--out_csv", type=str, default=CFG.PRED_CSV)
    p_pred.add_argument("--test_root", type=str, default=CFG.TEST_ROOT)
    p_pred.add_argument("--test_json_dir", type=str, default=CFG.TEST_JSON_DIR)
    p_pred.add_argument("--test_jpg_dir", type=str, default=CFG.TEST_JPG_DIR)

    p_fit = sub.add_parser("fit")
    add_common(p_fit)
    p_fit.add_argument("--epochs", type=int, default=CFG.EPOCHS)
    p_fit.add_argument("--lr", type=float, default=CFG.LEARNING_RATE)
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
    p_fit.add_argument("--resume_ckpt", type=str, default=None)

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
