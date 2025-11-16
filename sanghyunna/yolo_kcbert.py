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

# --- [★ 핵심 수정 1] ---
# Hugging Face 및 doclayout 라이브러리 임포트
try:
    from transformers import AutoTokenizer, AutoModel
    from doclayout_yolo import YOLOv10
    _NEW_BACKBONES_OK = True
except ImportError as e:
    print(f"Error: {e}. Please install 'transformers' and 'doclayout-yolo'.")
    _NEW_BACKBONES_OK = False
# --- [수정 완료] ---

# torchvision은 이제 사용하지 않으므로 체크 코드 제거
_BACKBONE_OK = _NEW_BACKBONES_OK


class CFG:
    # --- [★ 핵심 수정 2: CFG] ---
    # Core
    TEXT_MODEL_NAME: str = "beomi/KcELECTRA-base"
    # KcELECTRA-base의 히든 차원 (변경 불가)
    DIM: int = 768
    # YOLOv8 백본의 기본 입력 크기 (512 -> 640으로 상향 권장)
    IMG_SIZE: int = 640
    # --- [수정 완료] ---

    EPOCHS: int = 20
    LEARNING_RATE: float = 1e-4
    BATCH_SIZE: int = 32 # 64 -> 32 (모델이 커졌으므로 줄임)
    SEED: int = 42
    NUM_WORKERS: int = 16
    NO_PRETRAIN: bool = False  # True → disable *ImageNet* weights (YOLO/BERT는 항상 pretrained)

    CROSS_ATTN_DIM: Optional[int] = None  # fallback to DIM when not set
    CROSS_ATTN_LAYERS: int = 1
    CROSS_ATTN_HEADS: int = 2

    # Paths (기존과 동일)
    CKPT_PATH: str = "./outputs_kcELECTRA_yolo/ckpt/cross_attn_vlm.pth"
    RESUME_CKPT_PATH: str = None
    EVAL_CSV: str = "./outputs_kcELECTRA_yolo/preds/eval_pred.csv"
    PRED_CSV: str = "./outputs_kcELECTRA_yolo/preds/test_pred.csv"
    SUBMISSION_ZIP: str = "./outputs_kcELECTRA_yolo/submission.zip"

    # Optional split-specific overrides (기존과 동일)
    TRAIN_ROOT: Optional[str] = "../data/train_valid/train"
    VAL_ROOT: Optional[str] = "../data/train_valid/valid"
    TEST_ROOT: Optional[str] = "../data/open/test"
    TRAIN_JSON_DIR: Optional[str] = None
    TRAIN_JPG_DIR: Optional[str] = None
    VAL_JSON_DIR: Optional[str] = None
    VAL_JPG_DIR: Optional[str] = None
    TEST_JSON_DIR: Optional[str] = None
    TEST_JPG_DIR: Optional[str] = None
    USE_VAL_AS_TRAIN: bool = False

# ==================================================================
# Data Loading Utilities (기존 코드 복원)
# ==================================================================
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
    print(f"[discover] Found {len(pairs)} pairs in {root_dir}: {pairs}")
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
    # (이 함수는 Vocab 클래스가 삭제되면서 더 이상 사용되지 않습니다)
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

# --- [★ 핵심 수정 3: Vocab 클래스 삭제] ---
# Hugging Face 토크나이저를 사용하므로 Vocab 클래스가 더 이상 필요 없습니다.
# (class Vocab: ... 코드 전체 삭제)
# --- [수정 완료] ---


# --- [★ 핵심 수정 4: UniDSet 클래스] ---
# Vocab/토크나이징 관련 로직을 모두 제거하고, 원본 텍스트만 반환합니다.
class UniDSet(Dataset):
    def __init__(self, json_files: List[str], jpg_dir: Optional[str] = None,
                 resize_to: Tuple[int, int] = (CFG.IMG_SIZE, CFG.IMG_SIZE),
                 supervised_only: bool = False):
        
        self.items: List[Dict[str, Any]] = []
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
                
                # 텍스트 합치기 (기존 코드와 동일)
                qtxt = str(a.get("visual_instruction", "")).strip()
                cname = a.get("class_name", "") or ""
                combined_query = f"{cname} {qtxt}".strip()

                bbox = a.get("bounding_box", None)
                if supervised_only and not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
                    continue

                base, _ = os.path.splitext(img_path)
                cache_path = base + self.cache_suffix

                self.items.append({
                    "json": jf,
                    "img": img_path,
                    "query_id": qid,
                    "query": combined_query,  # <-- 원본 텍스트 저장
                    "bbox": bbox,
                    "class_name": cname,
                    "cache_path": cache_path,
                })

        # Vocab 빌드 및 인코딩 로직 *모두 삭제*
        
        # 이미지 전처리 (기존과 동일)
        try:
            from torchvision import transforms as T
            self.tf = T.Compose([
                T.Resize(resize_to), 
                T.ToTensor(),
                # KcELECTRA/YOLO 백본은 자체 정규화가 있거나 ToTensor(0-1)로 충분
            ])
        except ImportError:
            self.tf = None

    def __len__(self):
        return len(self.items)

    @staticmethod
    def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
        arr = np.array(img).astype(np.float32) / 255.0
        if arr.ndim == 2: arr = np.stack([arr, arr, arr], axis=-1)
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        it = self.items[idx]
        cache_path = it.get("cache_path", None)
        img_t: Optional[torch.Tensor] = None
        W: int
        H: int

        # 1) 캐시 로드 (기존과 동일)
        if cache_path is not None and os.path.isfile(cache_path):
            try:
                cached = torch.load(cache_path, map_location="cpu", weights_only=False)
                if isinstance(cached, dict) and "image" in cached and "orig_size" in cached:
                    img_t = cached["image"]
                    W, H = cached["orig_size"]
                else: raise ValueError("unexpected cache format")
            except Exception as e:
                print(f"[cache] failed to load {cache_path}: {e}")
                img_t = None
        
        # 2) 원본 로드 (기존과 동일)
        if img_t is None:
            img = Image.open(it["img"]).convert("RGB")
            W, H = img.size
            if self.tf is not None:
                img_t = self.tf(img)  # Resize + ToTensor
            else:
                try:
                    from torchvision.transforms.functional import resize, to_tensor
                    img_t = to_tensor(resize(img, self.resize_to, interpolation=Image.BILINEAR))
                except ImportError:
                    img = img.resize(self.resize_to, Image.BILINEAR)
                    img_t = self._pil_to_tensor(img)
            
            if cache_path is not None and not os.path.exists(cache_path):
                try:
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    torch.save({"image": img_t, "orig_size": (W, H)}, cache_path)
                except Exception as e:
                    print(f"[cache] failed to save {cache_path}: {e}")

        # 3) 반환 샘플 구성
        sample: Dict[str, Any] = {
            "image": img_t,
            "query_text": it["query"], # <-- 인코딩된 ID 대신 원본 텍스트 반환
            "query_id": it["query_id"],
            "orig_size": (W, H),
            "class_name": it["class_name"],
        }

        # Bbox 타겟 구성 (기존과 동일)
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
# --- [수정 완료] ---


# --- [★ 핵심 수정 5: collate_fn] ---
# 글로벌 토크나이저를 사용해 배치 단위로 실시간 토크나이징
try:
    TOKENIZER = AutoTokenizer.from_pretrained(CFG.TEXT_MODEL_NAME)
except Exception as e:
    print(f"Failed to load tokenizer '{CFG.TEXT_MODEL_NAME}'. {e}")
    TOKENIZER = None

def collate_fn(batch: List[Dict[str, Any]]):
    if TOKENIZER is None:
        raise RuntimeError(f"Tokenizer {CFG.TEXT_MODEL_NAME} not initialized.")

    # 1. 이미지, 타겟, 메타데이터 분리 (기존과 유사)
    imgs = torch.stack([b["image"] for b in batch], dim=0)
    targets = []
    meta = []
    query_texts = [] # 텍스트만 따로 리스트로 모음
    
    for b in batch:
        targets.append(b["target"])
        query_texts.append(b["query_text"])
        meta.append({
            "query_id": b["query_id"],
            "query_text": b["query_text"],
            "orig_size": b["orig_size"],
            "class_name": b["class_name"],
        })

    # 2. Hugging Face 토크나이저로 배치 처리
    # (padding, truncation, tensor 변환 자동)
    batch_token_dict = TOKENIZER(
        query_texts,
        padding=True,
        truncation=True,
        max_length=64, # 쿼리 최대 길이 (조정 가능)
        return_tensors="pt"
    )

    # 3. 타겟 스태킹 (기존과 동일)
    stacked_targets = None
    if all(t is not None for t in targets) and len(targets) > 0:
        stacked_targets = torch.stack(targets, dim=0)
        
    # 4. (imgs, 토큰 딕셔너리, 타겟, 메타, 스택된 타겟) 반환
    return imgs, batch_token_dict, targets, meta, stacked_targets
# --- [수정 완료] ---


# --- [★ 핵심 수정 6: TextEncoder] ---
# GRU -> KcELECTRA (BERT)
class TextEncoder(nn.Module):
    def __init__(self, emb_dim: int = CFG.DIM, hidden: int = CFG.DIM):
        super().__init__()
        # 1. KcELECTRA-base 모델 로드
        self.bert = AutoModel.from_pretrained(CFG.TEXT_MODEL_NAME)
        
        # 2. BERT 출력 차원 확인
        bert_dim = self.bert.config.hidden_size # (e.g., 768)
        
        # 3. (필요시) 최종 쿼리 벡터 차원(DIM)으로 프로젝션
        #    CFG.DIM이 768이므로 Identity
        if bert_dim == emb_dim:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Linear(bert_dim, emb_dim)
            
        print(f"[TextEncoder] Loaded '{CFG.TEXT_MODEL_NAME}' (dim={bert_dim}) -> proj to {emb_dim}")

    def forward(self, batch_token_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        # 1. 토크나이저가 만든 딕셔너리를 그대로 BERT에 전달
        outputs = self.bert(**batch_token_dict)
        
        # 2. [CLS] 토큰의 임베딩을 쿼리 벡터로 사용
        # outputs.last_hidden_state shape: (Batch, SeqLen, BertDim)
        cls_token = outputs.last_hidden_state[:, 0, :] # (Batch, BertDim)
        
        # 3. 최종 쿼리 벡터 q 생성
        q = self.proj(cls_token) # (Batch, DIM)
        return q
# --- [수정 완료] ---


# (TinyCNN은 더 이상 사용되지 않지만, 호환성을 위해 남겨둘 수 있음)
class TinyCNN(nn.Module):
    def __init__(self, out_dim: int = CFG.DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, out_dim, 3, stride=2, padding=1), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)


# --- [★ 핵심 수정 7: ImageEncoder] ---
# ResNet -> doclayout-yolo (YOLOv8) 백본

class ImageEncoder(nn.Module):
    def __init__(self, out_dim: int = CFG.DIM, pretrained: bool = True, img_size: int = CFG.IMG_SIZE):
        super().__init__()
        
        if not _NEW_BACKBONES_OK:
            raise ImportError("doclayout-yolo or transformers not found. Cannot create ImageEncoder.")
        
        if not pretrained and not CFG.NO_PRETRAIN:
            print("[Warn] 'pretrained=False' passed, but doclayout-yolo is always pretrained. Loading pretrained.")
        
        # 1. doclayout.load_model()로 YOLOv8 모델 로드
        try:
            yolo_wrapper = YOLOv10("./doclayout_yolo_docstructbench_imgsz1024.pt")
        except Exception as e:
            raise RuntimeError(f"Failed to load 'doclayout-yolo'. Is it installed? {e}")
            
        # 2. YOLOv8의 nn.Module (DetectionModel)에 접근
        self.backbone_full = yolo_wrapper.model.model
        
        # 3. YOLOv8의 백본(backbone) 부분만 사용 (Head 제외)
        #    DetectionModel.model은 [0]~[8]가 백본, [10+]가 헤드
        self.backbone_modules = self.backbone_full[:9]
        
        # 4. 백본의 출력 차원을 동적으로 확인
        target_indices = (4, 6, 8)  # (P3, P4, P5)에 해당하는 backbone layer index
        self.backbone_feature_indices = target_indices
        feature_channels = {}
        backbone_device = next(self.backbone_full.parameters()).device
        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size, img_size, device=backbone_device)
            y = dummy
            for i, m in enumerate(self.backbone_modules):
                y = m(y)
                if i in target_indices:
                    feature_channels[i] = y.shape[1]
        missing = [idx for idx in target_indices if idx not in feature_channels]
        if missing:
            raise RuntimeError(f"Failed to capture YOLO backbone feature dims at indices: {missing}")
        c3_channels = feature_channels[target_indices[0]]
        c4_channels = feature_channels[target_indices[1]]
        c5_channels = feature_channels[target_indices[2]]
        
        # 5. [★핵심 수정★] 3개의 피처맵을 모두 out_dim (768)으로 프로젝션
        self.proj_p3 = nn.Conv2d(c3_channels, out_dim, kernel_size=1)
        self.proj_p4 = nn.Conv2d(c4_channels, out_dim, kernel_size=1)
        self.proj_p5 = nn.Conv2d(c5_channels, out_dim, kernel_size=1)
        # --- [수정 완료] ---
        
        print(
            "[ImageEncoder] Loaded 'doclayout-yolo' (YOLOv8-L). "
            f"Using P3({c3_channels}ch), P4({c4_channels}ch), P5({c5_channels}ch) features."
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # x shape: (B, 3, 640, 640)
        
        # 1. YOLOv8 백본의 forward 실행
        p3, p4, p5 = None, None, None
        idx_p3, idx_p4, idx_p5 = self.backbone_feature_indices
        y = x
        for i, m in enumerate(self.backbone_modules):
            y = m(y)
            if i == idx_p3: p3 = y  # (B, C3, 80, 80)
            if i == idx_p4: p4 = y  # (B, C4, 40, 40)
            if i == idx_p5: p5 = y  # (B, C5, 20, 20)
        
        # 2. [★핵심 수정★] 3개의 맵을 모두 프로젝션하여 리스트로 반환
        fmap_p3 = self.proj_p3(p3) # (B, 768, 80, 80)
        fmap_p4 = self.proj_p4(p4) # (B, 768, 40, 40)
        fmap_p5 = self.proj_p5(p5) # (B, 768, 20, 20)
        
        return [fmap_p3, fmap_p4, fmap_p5]
        # --- [수정 완료] ---


# --- [★ 핵심 수정 8: CrossAttnVLM] ---
# vocab_size 제거 및 forward 시그니처 변경

class CrossAttentionLayer(nn.Module):
    def __init__(self, embed_dim: int, attn_dim: int, heads: int):
        super().__init__()
        if attn_dim <= 0: raise ValueError("cross_attn_dim must be positive")
        if heads <= 0: raise ValueError("cross_attn_heads must be positive")
        if attn_dim % heads != 0: raise ValueError("cross_attn_dim must be divisible by cross_attn_heads")
        
        self.embed_dim = embed_dim
        self.attn_dim = attn_dim
        self.heads = heads
        self.head_dim = attn_dim // heads
        
        self.q_proj = nn.Linear(embed_dim, attn_dim)
        
        # --- [★핵심 수정★] ---
        # Conv2d(embed_dim, ...) -> Linear(embed_dim, ...)
        # 4D 맵(B,D,H,W) 대신 3D 텐서(B,N,D)를 처리
        self.k_proj = nn.Linear(embed_dim, attn_dim)
        self.v_proj = nn.Linear(embed_dim, attn_dim)
        # --- [수정 완료] ---
        
        self.out_proj = nn.Linear(attn_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, q_vec: torch.Tensor, fmap_permuted: torch.Tensor) -> torch.Tensor:
        # q_vec: (B, D)
        # fmap_permuted: (B, N_all, D)
        
        B, N_all, D = fmap_permuted.shape
        
        # (B, D) -> (B, H, Dh)
        q = self.q_proj(q_vec).view(B, self.heads, self.head_dim)

        # --- [★핵심 수정★] ---
        # (B, N_all, D) -> (B, N_all, Attn_D) -> (B, N_all, H, Dh) -> (B, H, Dh, N_all)
        K = self.k_proj(fmap_permuted).view(B, N_all, self.heads, self.head_dim).permute(0, 2, 3, 1)
        V = self.v_proj(fmap_permuted).view(B, N_all, self.heads, self.head_dim).permute(0, 2, 3, 1)
        # --- [수정 완료] ---

        # (B, H, N_all)
        attn = torch.einsum("bhd,bhdn->bhn", q, K) / math.sqrt(self.head_dim)
        attn = torch.softmax(attn, dim=-1)
        
        # (B, H, Dh) -> (B, D)
        ctx = torch.einsum("bhn,bhdn->bhd", attn, V).reshape(B, self.attn_dim)
        out = self.out_proj(ctx)
        
        return self.norm(q_vec + out) # Residual + Norm

class CrossAttentionBBox(nn.Module):
    def __init__(self, embed_dim: int = CFG.DIM, attn_dim: Optional[int] = None,
                 num_layers: int = CFG.CROSS_ATTN_LAYERS, num_heads: int = CFG.CROSS_ATTN_HEADS):
        super().__init__()
        attn_dim = attn_dim if attn_dim is not None else embed_dim
        num_layers = max(1, num_layers)
        num_heads = max(1, num_heads)
        self.embed_dim = embed_dim
        self.attn_dim = attn_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        # CrossAttentionLayer는 이제 Linear K,V 프로젝션을 사용합니다.
        self.layers = nn.ModuleList([
            CrossAttentionLayer(embed_dim, attn_dim, num_heads) for _ in range(num_layers)
        ])
        
        self.bbox_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, 4)  # (cx, cy, w, h) normalized via sigmoid
        )

    def forward(self, q_vec: torch.Tensor, fmap_list: List[torch.Tensor]) -> torch.Tensor:
        
        # --- [★핵심 수정★] ---
        # fmap_list: [(B,D,80,80), (B,D,40,40), (B,D,20,20)]
        
        # 1. 모든 맵을 1D 시퀀스로 펼침 (B, D, N_i)
        fmap_flat_list = [f.flatten(2) for f in fmap_list]
        
        # 2. 하나의 거대한 시퀀스로 합침 (B, D, N_all)
        # N_all = 80*80 + 40*40 + 20*20 = 6400 + 1600 + 400 = 8400
        fmap_all_flat = torch.cat(fmap_flat_list, dim=2)
        
        # 3. Linear 레이어를 위해 (B, N_all, D) 형태로 축을 변경(permute)
        fmap_permuted = fmap_all_flat.permute(0, 2, 1)
        # --- [수정 완료] ---

        q = q_vec
        for layer in self.layers:
            # --- [★핵심 수정★] ---
            # (B, N_all, D) 텐서를 어텐션 레이어에 전달
            q = layer(q, fmap_permuted)
            # --- [수정 완료] ---
            
        pred = self.bbox_head(q)
        pred = torch.sigmoid(pred)
        return pred

class CrossAttnVLM(nn.Module):
    def __init__(self, dim: int = CFG.DIM, pretrained_backbone: bool = True,
                 img_size: int = CFG.IMG_SIZE, cross_attn_dim: Optional[int] = CFG.CROSS_ATTN_DIM,
                 cross_attn_layers: int = CFG.CROSS_ATTN_LAYERS, cross_attn_heads: int = CFG.CROSS_ATTN_HEADS):
        super().__init__()
        attn_dim = cross_attn_dim if cross_attn_dim is not None else dim
        layers = max(1, cross_attn_layers)
        heads = max(1, cross_attn_heads)
        if attn_dim % heads != 0:
            raise ValueError(f"cross_attn_dim ({attn_dim}) must be divisible by cross_attn_heads ({heads})")

        # 1. vocab_size가 필요 없는 새 인코더 사용
        self.txt = TextEncoder(emb_dim=dim, hidden=dim)
        self.img = ImageEncoder(out_dim=dim, pretrained=pretrained_backbone, img_size=img_size)
        
        self.head = CrossAttentionBBox(embed_dim=dim, attn_dim=attn_dim,
                                       num_layers=layers, num_heads=heads)
        self.cross_attn_dim = attn_dim
        self.cross_attn_layers = layers
        self.cross_attn_heads = heads

    def forward(self, images: torch.Tensor, batch_token_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        # 1. 토큰 딕셔너리를 모델과 동일한 디바이스로 이동
        batch_token_dict = {k: v.to(images.device) for k, v in batch_token_dict.items()}
        
        # 2. 새 시그니처로 각 인코더 호출
        q = self.txt(batch_token_dict)             # (B, D)
        fmap = self.img(images)                   # (B, D, H', W')
        pred_norm = self.head(q, fmap)            # (B, 4) in [0,1]
        return pred_norm
# --- [수정 완료] ---


# ==================================================================
# Metrics and Prediction Export
# (iou_xywh_pixel / batch_iou_norm / export_predictions)
# ==================================================================

# (기존 코드 복원)
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
    # (기존 코드와 동일)
    px1 = pred[:, 0] - pred[:, 2] / 2.0; py1 = pred[:, 1] - pred[:, 3] / 2.0
    px2 = pred[:, 0] + pred[:, 2] / 2.0; py2 = pred[:, 1] + pred[:, 3] / 2.0
    gx1 = tgt[:, 0] - tgt[:, 2] / 2.0; gy1 = tgt[:, 1] - tgt[:, 3] / 2.0
    gx2 = tgt[:, 0] + tgt[:, 2] / 2.0; gy2 = tgt[:, 1] + tgt[:, 3] / 2.0
    ix1 = torch.max(px1, gx1); iy1 = torch.max(py1, gy1)
    ix2 = torch.min(px2, gx2); iy2 = torch.min(py2, gy2)
    inter_w = (ix2 - ix1).clamp(min=0); inter_h = (iy2 - iy1).clamp(min=0)
    inter = inter_w * inter_h
    area_p = (px2 - px1).clamp(min=0) * (py2 - py1).clamp(min=0)
    area_g = (gx2 - gx1).clamp(min=0) * (gy2 - gy1).clamp(min=0)
    union = (area_p + area_g - inter).clamp(min=1e-6)
    iou = inter / union
    return float(iou.mean().item())


# --- [★ 핵심 수정 9: export_predictions] ---
# 루프 변수 변경 (ids, lens -> batch_token_dict)
def export_predictions(model: nn.Module, loader: DataLoader, out_csv: str,
                       device: torch.device, compute_iou: bool = False,
                       progress_desc: str = "predict") -> Optional[float]:
    rows = []
    ious: List[float] = []
    was_training = model.training
    model.eval()

    with torch.no_grad():
        # 1. 루프 변수명 변경
        iterator = tqdm(loader, desc=progress_desc, leave=False, dynamic_ncols=True)
        for imgs, batch_token_dict, targets, meta, _ in iterator:
            
            imgs = imgs.to(device, non_blocking=True)
            # batch_token_dict는 모델 내부에서 디바이스로 이동
            
            # 2. 모델 호출 방식 변경
            pred = model(imgs, batch_token_dict)
            
            # 3. 이후 로직은 (기존 코드와 동일)
            for i in range(imgs.size(0)):
                W, H = meta[i]["orig_size"]
                cx, cy, nw, nh = [float(v) for v in pred[i].cpu().numpy().tolist()]
                x = (cx - nw / 2.0) * W; y = (cy - nh / 2.0) * H
                w = nw * W; h = nh * H
                rows.append({
                    "query_id": meta[i]["query_id"], "query_text": meta[i]["query_text"],
                    "pred_x": x, "pred_y": y, "pred_w": w, "pred_h": h,
                })
                if compute_iou and targets[i] is not None:
                    gt = [float(v) for v in targets[i].numpy().tolist()]
                    gx = (gt[0] - gt[2] / 2.0) * W; gy = (gt[1] - gt[3] / 2.0) * H
                    gw = gt[2] * W; gh = gt[3] * H
                    # (기존 코드 복원) 독립된 iou_xywh_pixel 함수 호출
                    ious.append(iou_xywh_pixel([x, y, w, h], [gx, gy, gw, gh]))

    if out_csv:
        out_dir = os.path.dirname(out_csv)
        if out_dir: os.makedirs(out_dir, exist_ok=True)
        try:
            import pandas as pd
            df = pd.DataFrame(rows, columns=["query_id", "query_text", "pred_x", "pred_y", "pred_w", "pred_h"])
            df.to_csv(out_csv, index=False, encoding="utf-8-sig")
            print(f"[Saved] {out_csv}")
        except ImportError:
            print("[Warn] pandas not installed. Skipping CSV save.")

    miou: Optional[float] = None
    if compute_iou and ious: miou = float(np.mean(ious))
    elif compute_iou: print("[Eval] No ground-truth boxes available; IoU skipped.")
    
    if was_training: model.train()
    return miou
# --- [수정 완료] ---


# --- [★ 핵심 수정 10: make_loader] ---
# vocab 관련 로직 모두 제거
def make_loader(dir_pairs: List[Tuple[str, str]],
                batch_size: int = CFG.BATCH_SIZE, img_size: int = CFG.IMG_SIZE,
                num_workers: int = CFG.NUM_WORKERS, shuffle: bool = False):
    if not dir_pairs:
        raise ValueError("dir_pairs must contain at least one (json_dir, jpg_dir) tuple")

    datasets: List[Dataset] = []
    for json_dir, jpg_dir in dir_pairs:
        json_files = find_jsons(json_dir)
        subset = UniDSet(
            json_files,
            jpg_dir=jpg_dir,
            resize_to=(img_size, img_size),
            supervised_only=shuffle, # shuffle(학습)일 때만 supervised_only
        )
        if len(subset) == 0:
            print(f"[warn] No usable samples in {json_dir}; skipping")
            continue
        datasets.append(subset)

    if not datasets:
        raise RuntimeError("No datasets constructed from provided directories.")

    full_ds: Dataset = datasets[0] if len(datasets) == 1 else torch.utils.data.ConcatDataset(datasets)

    loader_kwargs = {
        "batch_size": batch_size, "shuffle": shuffle, "num_workers": num_workers,
        "collate_fn": collate_fn, "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0 and shuffle, # shuffle일 때만 persistent
    }
    if num_workers > 0: loader_kwargs["prefetch_factor"] = 8
    
    dl = DataLoader(full_ds, **loader_kwargs)
    
    # Vocab을 반환하지 않음
    return full_ds, dl
# --- [수정 완료] ---


# --- [★ 핵심 수정 11: save_checkpoint] ---
# vocab 인수 제거
def save_checkpoint(
    model: nn.Module,
    img_size: int,
    ckpt_path: str,
    dim: int,
    no_pretrain: bool,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    epoch: Optional[float] = None,
    best_miou: Optional[float] = None,
    patience_counter: Optional[int] = None,
    best_epoch: Optional[int] = None,
):
    os.makedirs(os.path.dirname(ckpt_path) or '.', exist_ok=True)
    cross_dim = getattr(model, "cross_attn_dim", dim)
    cross_layers = getattr(model, "cross_attn_layers", CFG.CROSS_ATTN_LAYERS)
    cross_heads = getattr(model, "cross_attn_heads", CFG.CROSS_ATTN_HEADS)
    
    ckpt: Dict[str, Any] = {
        "model_state": model.state_dict(),
        # "vocab_itos" 제거
        "dim": dim,
        "no_pretrain": no_pretrain,
        "img_size": img_size,
        "cross_attn_dim": cross_dim,
        "cross_attn_layers": cross_layers,
        "cross_attn_heads": cross_heads,
        "text_model_name": CFG.TEXT_MODEL_NAME, # 복원을 위해 모델 이름 저장
    }
    if optimizer is not None: ckpt["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None: ckpt["scheduler_state"] = scheduler.state_dict()
    if scaler is not None: ckpt["scaler_state"] = scaler.state_dict()
    if epoch is not None: ckpt["epoch"] = float(epoch)
    if best_miou is not None: ckpt["best_miou"] = float(best_miou)
    if patience_counter is not None: ckpt["patience_counter"] = int(patience_counter)
    if best_epoch is not None: ckpt["best_epoch"] = int(best_epoch)
    
    torch.save(ckpt, ckpt_path)
    print(f"[Saved] {ckpt_path}")
# --- [수정 완료] ---


# --- [★ 핵심 수정 12: train_one_epoch] ---
# (기존 코드 복원 및 루프 변수 변경)
def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                    scaler: torch.amp.GradScaler, device: torch.device,
                    desc: Optional[str] = None, on_half_epoch=None) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_iou = 0.0
    total = 0
    num_batches = len(loader)
    half_point = num_batches // 2

    iterator = tqdm(loader, desc=desc or "train", leave=False, dynamic_ncols=True)
    # 1. 루프 변수명 변경 (ids, lens -> batch_token_dict)
    for step, (imgs, batch_token_dict, targets, meta, stacked_targets) in enumerate(iterator, start=1):
        
        imgs = imgs.to(device, non_blocking=True)
        # batch_token_dict는 모델 내부에서 디바이스로 이동
        
        if stacked_targets is not None:
            t = stacked_targets.to(device, non_blocking=True)
        else:
            # stack_targets가 None이 되는 경우 (e.g. batch_size=1) 대비
            valid_targets = [tar for tar in targets if tar is not None]
            if not valid_targets:
                continue # 학습할 타겟이 없는 배치
            t = torch.stack(valid_targets, dim=0).to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            # 2. 모델 호출 방식 변경
            pred = model(imgs, batch_token_dict)
            
            # (stacked_targets가 None일 때 pred와 t의 개수가 다를 수 있음)
            if stacked_targets is None:
                valid_idx = [i for i, tar in enumerate(targets) if tar is not None]
                if not valid_idx:
                    continue
                pred = pred[valid_idx] # 타겟이 있는 샘플의 예측만 선택

            loss = F.smooth_l1_loss(pred, t, reduction="mean")

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        B = t.size(0) # 실제 학습에 사용된 샘플 수 기준
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
# --- [수정 완료] ---


# (evaluate_loss는 fit_pipeline에서만 사용되므로 여기에 둡니다)
def evaluate_loss(model: nn.Module, loader: DataLoader, device: torch.device,
                  desc: Optional[str] = None) -> float:
    model.eval()
    running = 0.0
    total = 0
    with torch.no_grad():
        iterator = tqdm(loader, desc=desc or "eval", leave=False, dynamic_ncols=True)
        for imgs, batch_token_dict, targets, meta, stacked_targets in iterator:
            valid_idx = [i for i, tar in enumerate(targets) if tar is not None]
            if not valid_idx:
                continue
            
            imgs = imgs.to(device, non_blocking=True)
            pred = model(imgs, batch_token_dict)
            pred_sel = pred[valid_idx]
            
            if stacked_targets is not None:
                tgt_sel = stacked_targets[valid_idx].to(device, non_blocking=True)
            else:
                tgt_sel = torch.stack([targets[i] for i in valid_idx], dim=0).to(device)
                
            loss = F.smooth_l1_loss(pred_sel, tgt_sel, reduction="mean")
            running += float(loss.item()) * len(valid_idx)
            total += len(valid_idx)
    return running / max(1, total)


# --- [★ 핵심 수정 13: _load_model_from_ckpt] ---
# vocab 로직 제거
def _load_model_from_ckpt(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    
    # Vocab 로드 로직 제거
    
    # 모델 이름/크기 로드
    text_model = ckpt.get("text_model_name", CFG.TEXT_MODEL_NAME)
    if text_model != CFG.TEXT_MODEL_NAME:
        # CFG를 덮어쓰기 (중요: 글로벌 변수 수정)
        print(f"[Warn] Checkpoint was trained with {text_model}. Overriding CFG.TEXT_MODEL_NAME.")
        CFG.TEXT_MODEL_NAME = text_model
        # 글로벌 토크나이저도 다시 로드
        global TOKENIZER
        TOKENIZER = AutoTokenizer.from_pretrained(CFG.TEXT_MODEL_NAME)

    
    cross_dim = ckpt.get("cross_attn_dim", ckpt.get("dim", CFG.DIM))
    cross_layers = ckpt.get("cross_attn_layers", ckpt.get("num_cross_attn", CFG.CROSS_ATTN_LAYERS))
    cross_heads = ckpt.get("cross_attn_heads", CFG.CROSS_ATTN_HEADS)
    img_size = ckpt.get("img_size", CFG.IMG_SIZE)

    model = CrossAttnVLM(
        # vocab_size 제거
        dim=ckpt["dim"],
        pretrained_backbone=not ckpt.get("no_pretrain", False),
        img_size=img_size,
        cross_attn_dim=cross_dim,
        cross_attn_layers=cross_layers,
        cross_attn_heads=cross_heads,
    ).to(device)
    
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    
    train_state = {
        "optimizer_state": ckpt.get("optimizer_state"),
        "scheduler_state": ckpt.get("scheduler_state"),
        "scaler_state": ckpt.get("scaler_state"),
        "epoch": ckpt.get("epoch"),
        "best_miou": ckpt.get("best_miou"),
        "patience_counter": ckpt.get("patience_counter"),
        "best_epoch": ckpt.get("best_epoch"),
    }
    
    # vocab 반환 제거
    return model, img_size, train_state
# --- [수정 완료] ---


# (기존 코드)
def infer_start_epoch(train_state: Optional[Dict[str, Any]], default_start: int = 1) -> int:
    state = train_state or {}
    marker = state.get("epoch")
    if marker is None: return default_start
    try: marker = float(marker)
    except (TypeError, ValueError): return default_start
    return max(default_start, int(math.floor(marker)) + 1)


# --- [★ 핵심 수정 17: train_loop 복원 및 수정] ---
# (원본 `train_loop` 로직 기반으로 수정)
def train_loop(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_pairs = resolve_dir_pairs(
        getattr(args, "train_json_dir", None),
        getattr(args, "train_jpg_dir", None),
        getattr(args, "train_root", None),
        "train",
    )

    resume_ckpt = getattr(args, "resume_ckpt", None)
    train_state: Dict[str, Any] = {}

    if resume_ckpt:
        # 1. vocab 반환 받지 않음
        model, used_img_size, train_state = _load_model_from_ckpt(resume_ckpt, device)
        ckpt_dim = model.txt.bert.config.hidden_size # BERT에서 실제 dim 읽기

        # 2. vocab 전달/빌드 안 함
        train_ds, train_dl = make_loader(
            train_pairs, batch_size=args.batch_size,
            img_size=used_img_size, num_workers=args.num_workers, shuffle=True,
        )
        print(f"[Resume] Loaded checkpoint from {resume_ckpt}")
    else:
        # 3. 처음부터 학습
        used_img_size = args.img_size
        train_ds, train_dl = make_loader(
            train_pairs, batch_size=args.batch_size,
            img_size=used_img_size, num_workers=args.num_workers, shuffle=True,
        )
        # 4. vocab_size 없이 모델 생성
        model = CrossAttnVLM(
            dim=args.dim,
            pretrained_backbone=not args.no_pretrain,
            img_size=used_img_size,
            cross_attn_dim=args.cross_attn_dim,
            cross_attn_layers=args.cross_attn_layers,
            cross_attn_heads=args.cross_attn_heads,
        ).to(device)
        ckpt_dim = args.dim

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    
    if train_state.get("optimizer_state"): optimizer.load_state_dict(train_state["optimizer_state"])
    if train_state.get("scheduler_state"): scheduler.load_state_dict(train_state["scheduler_state"])
    if train_state.get("scaler_state"): scaler.load_state_dict(train_state["scaler_state"])

    if hasattr(train_dl, "dataset") and hasattr(train_dl.dataset, "__len__"):
        total_samples = len(train_dl.dataset)  # type: ignore[arg-type]
    else:
        total_samples = len(train_dl) * args.batch_size

    start_epoch = infer_start_epoch(train_state, 1)
    last_epoch = start_epoch - 1
    
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running = 0.0
        batch_iter = tqdm(
            train_dl,
            desc=f"train epoch {epoch}/{args.epochs}",
            leave=False,
            dynamic_ncols=True,
        )
        # 5. 루프 변수명 변경 (ids, lens -> batch_token_dict)
        for imgs, batch_token_dict, targets, meta, stacked_targets in batch_iter:
            imgs = imgs.to(device, non_blocking=True)
            
            if stacked_targets is not None:
                t = stacked_targets.to(device, non_blocking=True)
            else:
                valid_targets = [tar for tar in targets if tar is not None]
                if not valid_targets: continue
                t = torch.stack(valid_targets, dim=0).to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                # 6. 모델 호출 방식 변경
                pred = model(imgs, batch_token_dict)
                
                if stacked_targets is None:
                    valid_idx = [i for i, tar in enumerate(targets) if tar is not None]
                    if not valid_idx: continue
                    pred = pred[valid_idx]

                loss = F.smooth_l1_loss(pred, t, reduction="mean")
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.item()) * t.size(0)

        scheduler.step()
        last_epoch = epoch
        avg = running / max(1, total_samples)
        print(f"[Epoch {epoch}/{args.epochs}] loss={avg:.4f}  lr={scheduler.get_last_lr()[0]:.6f}")

    # 7. vocab 없이 체크포인트 저장
    save_checkpoint(
        model, used_img_size, args.save_ckpt, ckpt_dim, args.no_pretrain,
        optimizer=optimizer, scheduler=scheduler, scaler=scaler, epoch=last_epoch,
    )
# --- [수정 완료] ---


# --- [★ 핵심 수정 14: fit_pipeline (메인 로직)] ---
# (이 코드는 이미 이전 턴에서 수정 완료됨 - vocab 로직 제거)
def fit_pipeline(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_pairs = resolve_dir_pairs(args.train_json_dir, args.train_jpg_dir, args.train_root, "train")
    val_pairs = resolve_dir_pairs(args.val_json_dir, args.val_jpg_dir, args.val_root, "val")
    test_pairs = resolve_dir_pairs(args.test_json_dir, args.test_jpg_dir, args.test_root, "test")

    use_val_as_train = bool(getattr(args, "use_val_as_train", CFG.USE_VAL_AS_TRAIN))
    combined_train_pairs = train_pairs + val_pairs if use_val_as_train else train_pairs

    resume_ckpt = getattr(args, "resume_ckpt", None)
    train_state: Dict[str, Any] = {}

    if resume_ckpt:
        # 1. vocab 반환 받지 않음
        model, used_img_size, train_state = _load_model_from_ckpt(resume_ckpt, device)
        ckpt_dim = model.txt.bert.config.hidden_size # BERT에서 실제 dim 읽기

        # 2. vocab 전달/빌드 안 함
        train_ds, train_dl = make_loader(
            combined_train_pairs, batch_size=args.batch_size,
            img_size=used_img_size, num_workers=args.num_workers, shuffle=True,
        )
        val_ds, val_dl = make_loader(
            val_pairs, batch_size=args.batch_size,
            img_size=used_img_size, num_workers=args.num_workers, shuffle=False,
        )
        print(f"[Resume] Loaded checkpoint from {resume_ckpt}")
    else:
        # 3. 처음부터 학습
        used_img_size = args.img_size
        train_ds, train_dl = make_loader(
            combined_train_pairs, batch_size=args.batch_size,
            img_size=used_img_size, num_workers=args.num_workers, shuffle=True,
        )
        val_ds, val_dl = make_loader(
            val_pairs, batch_size=args.batch_size,
            img_size=used_img_size, num_workers=args.num_workers, shuffle=False,
        )

        # 4. vocab_size 없이 모델 생성
        model = CrossAttnVLM(
            dim=args.dim,
            pretrained_backbone=not args.no_pretrain,
            img_size=used_img_size,
            cross_attn_dim=args.cross_attn_dim,
            cross_attn_layers=args.cross_attn_layers,
            cross_attn_heads=args.cross_attn_heads,
        ).to(device)
        ckpt_dim = args.dim

    # (이후 로직은 기존 코드와 거의 동일)
    
    eval_loader_kwargs = {
        "batch_size": args.batch_size, "shuffle": False, "num_workers": args.num_workers,
        "collate_fn": collate_fn, "pin_memory": torch.cuda.is_available(),
        "persistent_workers": args.num_workers > 0,
    }
    if args.num_workers > 0: eval_loader_kwargs["prefetch_factor"] = 8
    val_eval_dl = DataLoader(val_ds, **eval_loader_kwargs)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    
    if train_state.get("optimizer_state"): optimizer.load_state_dict(train_state["optimizer_state"])
    if train_state.get("scheduler_state"): scheduler.load_state_dict(train_state["scheduler_state"])
    if train_state.get("scaler_state"): scaler.load_state_dict(train_state["scaler_state"])

    best_miou = float(train_state.get("best_miou") or 0.0)
    best_epoch = int(train_state.get("best_epoch") or 0)
    patience_ctr = int(train_state.get("patience_counter") or 0)
    ckpt_base, ckpt_ext = os.path.splitext(args.save_ckpt)

    def eval_and_ckpt(tag: str, train_iou: Optional[float] = None) -> Tuple[Optional[float], str]:
        val_miou = export_predictions(
            model, val_eval_dl, out_csv="", device=device,
            compute_iou=True, progress_desc=f"val miou {tag}",
        )
        if val_miou is not None:
            if train_iou is not None: print(f"[mIoU {tag}] train={train_iou:.4f}  val={val_miou:.4f}")
            else: print(f"[mIoU {tag}] val={val_miou:.4f}")
        ckpt_path = f"{ckpt_base}_{tag}{ckpt_ext}"
        return val_miou, ckpt_path

    start_epoch = infer_start_epoch(train_state, 1)
    
    for epoch in range(start_epoch, args.epochs + 1):
        def half_cb(cur_train_iou, ep=epoch):
            tag = f"ep{ep - 0.5:.1f}"
            _, ckpt_path = eval_and_ckpt(tag, train_iou=cur_train_iou)
            # 5. vocab 없이 체크포인트 저장
            save_checkpoint(
                model, used_img_size, ckpt_path, ckpt_dim, args.no_pretrain,
                optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                epoch=ep - 0.5, best_miou=best_miou,
                patience_counter=patience_ctr, best_epoch=best_epoch,
            )

        train_loss, train_iou = train_one_epoch(
            model, train_dl, optimizer, scaler, device,
            desc=f"train {epoch}/{args.epochs}", on_half_epoch=half_cb,
        )
        scheduler.step()
        
        # (val_loss는 참고용으로 계산 - 기존 코드 복원)
        val_loss = evaluate_loss(model, val_eval_dl, device, desc="val loss")
        print(
            f"[Epoch {epoch}/{args.epochs}] train_loss={train_loss:.4f} train_mIoU={train_iou:.4f} "
            f"val_loss={val_loss:.4f} lr={scheduler.get_last_lr()[0]:.6f}"
        )

        full_tag = f"ep{epoch:.1f}"
        val_miou, tag_ckpt_path = eval_and_ckpt(full_tag, train_iou=train_iou)
        
        should_stop = False
        if val_miou is not None and val_miou >= best_miou - args.early_stop_delta:
            if val_miou > best_miou + args.early_stop_delta:
                patience_ctr = 0
                print(f"[Best mIoU] New best: {val_miou:.4f} (was {best_miou:.4f})")
                best_epoch = epoch
            else:
                patience_ctr += 1
            best_miou = max(best_miou, val_miou)
            # 6. vocab 없이 체크포인트 저장 (best)
            save_checkpoint(
                model, used_img_size, args.save_ckpt, ckpt_dim, args.no_pretrain,
                optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                epoch=epoch, best_miou=best_miou,
                patience_counter=patience_ctr, best_epoch=best_epoch,
            )
        elif val_miou is None:
            print("[Warn] val_miou is None, skipping early stopping check.")
            patience_ctr += 1
        else:
            patience_ctr += 1
            print(f"[EarlyStop] mIoU not improved ({val_miou:.4f} < {best_miou:.4f}). Patience: {patience_ctr}/{args.early_stop_patience}")
            if patience_ctr >= args.early_stop_patience:
                print("[EarlyStop] patience exhausted; halting training")
                should_stop = True
        
        # 7. vocab 없이 체크포인트 저장 (epoch-end)
        save_checkpoint(
            model, used_img_size, tag_ckpt_path, ckpt_dim, args.no_pretrain,
            optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            epoch=epoch, best_miou=best_miou,
            patience_counter=patience_ctr, best_epoch=best_epoch,
        )
        if should_stop:
            break
            
    print(f"[Best] epoch={best_epoch} val_mIoU={best_miou:.4f}")

    # 8. 최종 평가/추론 (vocab 로직 제거)
    print(f"Loading best model from {args.save_ckpt} for final eval/pred...")
    model, best_img_size, _ = _load_model_from_ckpt(args.save_ckpt, device)

    _, val_eval_dl = make_loader(
        val_pairs, batch_size=args.batch_size,
        img_size=best_img_size, num_workers=args.num_workers, shuffle=False,
    )
    miou = export_predictions(model, val_eval_dl, args.eval_csv, device, compute_iou=True, progress_desc="eval preds")
    if miou is not None: print(f"[Final Eval] mIoU={miou:.4f}")

    _, test_dl = make_loader(
        test_pairs, batch_size=args.batch_size,
        img_size=best_img_size, num_workers=args.num_workers, shuffle=False,
    )
    export_predictions(model, test_dl, args.pred_csv, device, compute_iou=False, progress_desc="test preds")
# --- [수정 완료] ---


# --- [★ 핵심 수정 15: get_args (CFG 기본값 반영)] ---
# CFG의 변경된 기본값(DIM=768, BATCH_SIZE=32 등)을 사용하도록 수정
def get_args():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_common(p):
        p.add_argument("--batch_size", type=int, default=CFG.BATCH_SIZE,
                       help=f"Batch size (default: {CFG.BATCH_SIZE})")
        p.add_argument("--img_size", type=int, default=CFG.IMG_SIZE,
                       help=f"Image resize size (default: {CFG.IMG_SIZE})")
        p.add_argument("--dim", type=int, default=CFG.DIM,
                       help=f"Model dimension (default: {CFG.DIM}, from KcELECTRA)")
        p.add_argument("--num_workers", type=int, default=CFG.NUM_WORKERS,
                       help=f"Dataloader workers (default: {CFG.NUM_WORKERS})")
        p.add_argument("--cross_attn_dim", type=int, default=CFG.CROSS_ATTN_DIM,
                       help="Cross-attention projection dim (defaults to --dim when omitted)")
        p.add_argument("--cross_attn_layers", type=int, default=CFG.CROSS_ATTN_LAYERS,
                       help="Number of cross-attention refinement layers")
        p.add_argument("--cross_attn_heads", type=int, default=CFG.CROSS_ATTN_HEADS,
                       help="Number of attention heads per cross-attention layer")

    # train
    p_train = sub.add_parser("train", help="Train model without validation/early stopping")
    add_common(p_train)
    p_train.add_argument("--epochs", type=int, default=CFG.EPOCHS)
    p_train.add_argument("--lr", type=float, default=CFG.LEARNING_RATE)
    p_train.add_argument("--no_pretrain", action="store_true", help="Disable pretrained weights (YOLO/BERT always pretrained)")
    p_train.add_argument("--save_ckpt", type=str, default=CFG.CKPT_PATH)
    p_train.add_argument("--train_root", type=str, default=CFG.TRAIN_ROOT)
    p_train.add_argument("--train_json_dir", type=str, default=CFG.TRAIN_JSON_DIR)
    p_train.add_argument("--train_jpg_dir", type=str, default=CFG.TRAIN_JPG_DIR)
    p_train.add_argument("--resume_ckpt", type=str, default=CFG.RESUME_CKPT_PATH)

    # eval
    p_eval = sub.add_parser("eval", help="Evaluate model mIoU on validation set")
    add_common(p_eval)
    p_eval.add_argument("--ckpt", type=str, required=True, help="Path to .pth checkpoint")
    p_eval.add_argument("--out_csv", type=str, default=CFG.EVAL_CSV)
    p_eval.add_argument("--val_root", type=str, default=CFG.VAL_ROOT)
    p_eval.add_argument("--val_json_dir", type=str, default=CFG.VAL_JSON_DIR)
    p_eval.add_argument("--val_jpg_dir", type=str, default=CFG.VAL_JPG_DIR)

    # predict
    p_pred = sub.add_parser("predict", help="Run inference on test set")
    add_common(p_pred)
    p_pred.add_argument("--ckpt", type=str, required=True, help="Path to .pth checkpoint")
    p_pred.add_argument("--out_csv", type=str, default=CFG.PRED_CSV)
    p_pred.add_argument("--test_root", type=str, default=CFG.TEST_ROOT)
    p_pred.add_argument("--test_json_dir", type=str, default=CFG.TEST_JSON_DIR)
    p_pred.add_argument("--test_jpg_dir", type=str, default=CFG.TEST_JPG_DIR)

    # fit
    p_fit = sub.add_parser("fit", help="Train with validation, mIoU checks, and early stopping")
    add_common(p_fit)
    p_fit.add_argument("--epochs", type=int, default=CFG.EPOCHS)
    p_fit.add_argument("--lr", type=float, default=CFG.LEARNING_RATE)
    p_fit.add_argument("--no_pretrain", action="store_true")
    p_fit.add_argument("--save_ckpt", type=str, default=CFG.CKPT_PATH, help="Path to save *best* model")
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
    p_fit.add_argument("--resume_ckpt", type=str, default=CFG.RESUME_CKPT_PATH)
    p_fit.add_argument("--use_val_as_train", action="store_true", default=CFG.USE_VAL_AS_TRAIN)

    # submission
    p_zip = sub.add_parser("zip", help="Zip a CSV file for submission")
    p_zip.add_argument("--csv", type=str, required=True, help="Path to .csv file to zip")
    p_zip.add_argument("--out_zip", type=str, default=CFG.SUBMISSION_ZIP)

    return ap.parse_args()
# --- [수정 완료] ---


# --- [★ 핵심 수정 16: main (evaluate_loop, predict_loop 수정)] ---
# vocab 관련 로직 제거
def evaluate_loop(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 1. vocab 반환 받지 않음
    model, img_size, _ = _load_model_from_ckpt(args.ckpt, device)
    val_pairs = resolve_dir_pairs(args.val_json_dir, args.val_jpg_dir, args.val_root, "val")
    # 2. vocab 전달 안 함
    _, dl = make_loader(
        val_pairs, batch_size=args.batch_size, img_size=img_size,
        num_workers=args.num_workers, shuffle=False
    )
    miou = export_predictions(model, dl, args.out_csv, device, compute_iou=True, progress_desc="eval preds")
    if miou is not None: print(f"[Eval] mIoU={miou:.4f}")

def predict_loop(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 1. vocab 반환 받지 않음
    model, img_size, _ = _load_model_from_ckpt(args.ckpt, device)
    test_pairs = resolve_dir_pairs(args.test_json_dir, args.test_jpg_dir, args.test_root, "test")
    # 2. vocab 전달 안 함
    _, dl = make_loader(
        test_pairs, batch_size=args.batch_size, img_size=img_size,
        num_workers=args.num_workers, shuffle=False
    )
    export_predictions(model, dl, args.out_csv, device, compute_iou=False, progress_desc="test preds")


def main():
    if not _BACKBONE_OK:
        print("Required libraries (transformers, doclayout-yolo) not found. Exiting.")
        return
        
    seed_everything(CFG.SEED)
    args = get_args()

    if args.cmd == "train":
        train_loop(args) # 'train' 명령어 실행
    elif args.cmd == "fit":
        fit_pipeline(args) # 'fit' 명령어 실행
    elif args.cmd == "eval":
        evaluate_loop(args)
    elif args.cmd == "predict":
        predict_loop(args)
    elif args.cmd == "zip":
        zip_submission(args.csv, args.out_zip)
    else:
        raise ValueError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
