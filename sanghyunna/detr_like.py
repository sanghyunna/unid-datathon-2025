import argparse
import json
import math
import os
import random
from glob import glob
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError as exc:  # pragma: no cover - informative guard
    raise ImportError("Install transformers to run detr_like.py") from exc

try:
    from doclayout_yolo import YOLOv10
    _HAS_DOCLAYOUT = True
except Exception:
    YOLOv10 = None
    _HAS_DOCLAYOUT = False


class CFG:
    """Default configuration tuned for a lighter DETR-style model."""

    TEXT_MODEL: str = "beomi/KcELECTRA-base"
    DIM: int = 512  # project BERT (768) into this dimension
    IMG_SIZE: int = 512
    NUM_QUERIES: int = 10
    NUM_ENCODER_LAYERS: int = 2
    NUM_DECODER_LAYERS: int = 3
    NUM_HEADS: int = 4
    FFN_DIM: int = 1024
    DROPOUT: float = 0.1

    FREEZE_TEXT: bool = True
    USE_DOCLAYOUT: bool = False  # default to tiny CNN for stability/cost
    FREEZE_DOCLAYOUT: bool = True

    LR: float = 1e-4
    EPOCHS: int = 30
    BATCH_SIZE: int = 64
    NUM_WORKERS: int = 16

    TRAIN_ROOT: Optional[str] = "../data/train_valid/train"
    VAL_ROOT: Optional[str] = "../data/train_valid/valid"
    TRAIN_JSON_DIR: Optional[str] = None
    TRAIN_JPG_DIR: Optional[str] = None
    VAL_JSON_DIR: Optional[str] = None
    VAL_JPG_DIR: Optional[str] = None

    COST_BBOX: float = 1.0
    COST_GIOU: float = 0.5
    COST_OBJ: float = 0.25

    WEIGHT_BBOX: float = 2.0
    WEIGHT_GIOU: float = 1.0
    WEIGHT_OBJ: float = 1.0

    CKPT_PATH: str = "./outputs_kcELECTRA_yolo/ckpt/detr_like_best.pth"
    BAD_JSON_WHITELIST: str = "./bad_json_whitelist.txt"


# ---------------------------------------------------------------------------
# Utility helpers (JSON discovery + filtering)
# ---------------------------------------------------------------------------

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def discover_json_jpg_pairs(root_dir: str) -> List[Tuple[str, str]]:
    if not root_dir:
        raise ValueError("root_dir is required")
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(root_dir)

    child_dirs = sorted(
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    )

    suffixes = [
        "_json", "json", "_jpg", "jpg", "_jpeg", "jpeg",
        "_img", "img", "_image", "image", "_images", "images",
        "_query", "query", "_queries", "queries",
    ]

    def normalize_base(name: str) -> str:
        lower = name.lower()
        for suf in suffixes:
            if lower.endswith(suf):
                lower = lower[: -len(suf)]
                break
        return lower.rstrip("_-") or "root"

    def has_files(path: str, patterns: Sequence[str]) -> bool:
        return any(glob(os.path.join(path, patt)) for patt in patterns)

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

    buckets: Dict[str, Dict[str, Optional[str]]] = {}

    for name in child_dirs:
        path = os.path.join(root_dir, name)
        json_like = is_json_dir(name, path)
        img_like = is_image_dir(name, path)
        if not (json_like or img_like):
            continue
        base = normalize_base(name)
        entry = buckets.setdefault(base, {"json": None, "jpg": None})
        if json_like and entry["json"] is None:
            entry["json"] = path
        if img_like and entry["jpg"] is None:
            entry["jpg"] = path

    pairs = [(info["json"], info["jpg"]) for info in buckets.values() if info["json"] and info["jpg"]]
    if not pairs:
        raise ValueError(f"Could not match json/jpg dirs under {root_dir}")
    return pairs


def resolve_dir_pairs(json_dir: Optional[str], jpg_dir: Optional[str], root_dir: Optional[str]) -> List[Tuple[str, str]]:
    if root_dir:
        return discover_json_jpg_pairs(root_dir)
    if json_dir and jpg_dir:
        return [(json_dir, jpg_dir)]
    raise ValueError("Provide either root_dir or both json_dir/jpg_dir")


def find_json_files(json_dir: str) -> List[str]:
    if not os.path.isdir(json_dir):
        raise FileNotFoundError(json_dir)
    return sorted(glob(os.path.join(json_dir, "*.json")))


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_image_path(json_path: str, data: Dict[str, Any], jpg_dir: Optional[str]) -> str:
    src = data.get("source_data_info", {})
    jpg_name = src.get("source_data_name_jpg")
    if jpg_dir and jpg_name:
        candidate = os.path.join(jpg_dir, jpg_name)
        if os.path.exists(candidate):
            return candidate
    if jpg_name:
        maybe = json_path.replace(os.sep + "json" + os.sep, os.sep + "jpg" + os.sep)
        maybe = os.path.join(os.path.dirname(maybe), jpg_name)
        if os.path.exists(maybe):
            return maybe
    base = os.path.splitext(json_path)[0]
    fallback = base.replace("MI3", "MI2") + ".jpg"
    if os.path.exists(fallback):
        return fallback
    raise FileNotFoundError(f"Cannot locate jpg for {json_path}")


def is_visual_ann(ann: Dict[str, Any]) -> bool:
    cid = str(ann.get("class_id", ""))
    cname = str(ann.get("class_name", ""))
    query = str(ann.get("visual_instruction", "")).strip()
    looks_visual = cid.startswith("V") or any(token in cname for token in ["표", "차트", "graph", "table"])
    return looks_visual and bool(query)


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
    return bad


BAD_JSON = load_bad_json_whitelist(CFG.BAD_JSON_WHITELIST)


# ---------------------------------------------------------------------------
# Dataset + collate
# ---------------------------------------------------------------------------


class QueryDataset(Dataset):
    def __init__(
        self,
        json_files: List[str],
        jpg_dir: Optional[str],
        resize_to: int = CFG.IMG_SIZE,
        supervised_only: bool = False,
    ) -> None:
        self.items: List[Dict[str, Any]] = []
        self.resize_to = resize_to
        self.supervised_only = supervised_only
        self._build_items(json_files, jpg_dir)
        try:
            from torchvision import transforms as T

            self.tf = T.Compose([
                T.Resize((resize_to, resize_to)),
                T.ToTensor(),
            ])
        except Exception:
            self.tf = None

    def _build_items(self, json_files: List[str], jpg_dir: Optional[str]) -> None:
        for jf in json_files:
            jf_norm = os.path.normpath(jf)
            if jf_norm in BAD_JSON:
                continue
            data = read_json(jf)
            ann_list = data.get("learning_data_info", {}).get("annotation", [])
            img_path = get_image_path(jf, data, jpg_dir)
            for ann in ann_list:
                if not is_visual_ann(ann):
                    continue
                bbox = ann.get("bounding_box")
                if self.supervised_only and not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
                    continue
                cname = ann.get("class_name", "") or ""
                query_txt = f"{cname} {ann.get('visual_instruction', '')}".strip()
                self.items.append(
                    {
                        "img": img_path,
                        "query": query_txt,
                        "bbox": bbox,
                        "class_name": cname,
                    }
                )

    def __len__(self) -> int:
        return len(self.items)

    def _pil_to_tensor(self, img: Image.Image) -> torch.Tensor:
        arr = np.array(img).astype(np.float32) / 255.0
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        img = Image.open(item["img"]).convert("RGB")
        W, H = img.size
        if self.tf is not None:
            img_tensor = self.tf(img)
        else:  # pragma: no cover - torchvision fallback
            img = img.resize((self.resize_to, self.resize_to), Image.BILINEAR)
            img_tensor = self._pil_to_tensor(img)

        bbox = item["bbox"]
        target_tensor = None
        if bbox and isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            x, y, w, h = bbox
            cx = (x + w / 2.0) / W
            cy = (y + h / 2.0) / H
            tw = w / W
            th = h / H
            target_tensor = torch.tensor([cx, cy, tw, th], dtype=torch.float32)

        return {
            "image": img_tensor,
            "text": item["query"],
            "target": target_tensor,
            "orig_size": (W, H),
            "class_name": item["class_name"],
        }


def build_collate_fn(tokenizer: AutoTokenizer):
    def _collate(batch: List[Dict[str, Any]]):
        images = torch.stack([b["image"] for b in batch], dim=0)
        texts = [b["text"] for b in batch]
        token_batch = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )
        targets: List[Dict[str, torch.Tensor]] = []
        metas: List[Dict[str, Any]] = []
        for b in batch:
            if b["target"] is None:
                tgt = {"boxes": torch.empty(0, 4, dtype=torch.float32)}
            else:
                tgt = {"boxes": b["target"].unsqueeze(0)}
            targets.append(tgt)
            metas.append({"orig_size": b["orig_size"], "query": b["text"], "class_name": b["class_name"]})
        return images, token_batch, targets, metas

    return _collate


# ---------------------------------------------------------------------------
# Positional encoding utilities
# ---------------------------------------------------------------------------


def build_2d_sincos_position_embedding(h: int, w: int, dim: int, device: torch.device) -> torch.Tensor:
    if dim % 4 != 0:
        raise ValueError("dim must be divisible by 4 for 2D sincos PE")
    y_embed = torch.linspace(0, 1, steps=h, device=device)
    x_embed = torch.linspace(0, 1, steps=w, device=device)
    grid_y, grid_x = torch.meshgrid(y_embed, x_embed, indexing="ij")
    omega = torch.arange(dim // 4, device=device, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / (dim // 4)))
    out_y = torch.cat([torch.sin(grid_y[..., None] * omega), torch.cos(grid_y[..., None] * omega)], dim=-1)
    out_x = torch.cat([torch.sin(grid_x[..., None] * omega), torch.cos(grid_x[..., None] * omega)], dim=-1)
    pos = torch.cat([out_y, out_x], dim=-1)
    pos = pos.view(h * w, dim)
    return pos


# ---------------------------------------------------------------------------
# Encoders and decoder blocks
# ---------------------------------------------------------------------------


class TextEncoder(nn.Module):
    def __init__(self, dim: int = CFG.DIM, freeze: bool = CFG.FREEZE_TEXT):
        super().__init__()
        self.model = AutoModel.from_pretrained(CFG.TEXT_MODEL)
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
        bert_dim = self.model.config.hidden_size
        self.proj = nn.Linear(bert_dim, dim)

    def forward(self, tokens: Dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = self.model(**tokens)
        cls = outputs.last_hidden_state[:, 0, :]
        return self.proj(cls)


class TinyCNN(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, dim, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        y = self.net(x)
        return [y]


class ImageEncoder(nn.Module):
    def __init__(self, dim: int = CFG.DIM, use_doclayout: bool = CFG.USE_DOCLAYOUT,
                 freeze_backbone: bool = CFG.FREEZE_DOCLAYOUT, img_size: int = CFG.IMG_SIZE):
        super().__init__()
        self.use_doclayout = use_doclayout and _HAS_DOCLAYOUT
        self.dim = dim
        if self.use_doclayout:
            yolo = YOLOv10("./doclayout_yolo_docstructbench_imgsz1024.pt")
            self.backbone = yolo.model.model[:9]
            if freeze_backbone:
                for p in self.backbone.parameters():
                    p.requires_grad = False
            target_indices = (4, 6, 8)
            feature_dims: Dict[int, int] = {}
            device = next(self.backbone.parameters()).device
            with torch.no_grad():
                dummy = torch.zeros(1, 3, img_size, img_size, device=device)
                y = dummy
                for i, module in enumerate(self.backbone):
                    y = module(y)
                    if i in target_indices:
                        feature_dims[i] = y.shape[1]
            self.feature_layers = target_indices
            self.projections = nn.ModuleList([
                nn.Conv2d(feature_dims[idx], dim, kernel_size=1)
                for idx in target_indices
            ])
        else:
            self.backbone = TinyCNN(dim)
            self.feature_layers = (0,)
            self.projections = nn.ModuleList([nn.Identity()])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats: List[torch.Tensor] = []
        if self.use_doclayout:
            y = x
            project_idx = 0
            for idx, module in enumerate(self.backbone):
                y = module(y)
                if idx == self.feature_layers[project_idx]:
                    feats.append(self.projections[project_idx](y))
                    project_idx += 1
                    if project_idx >= len(self.feature_layers):
                        break
        else:
            feats = [self.projections[0](self.backbone(x)[0])]

        tokens: List[torch.Tensor] = []
        pos_list: List[torch.Tensor] = []
        for feat in feats:
            B, C, H, W = feat.shape
            token = feat.flatten(2).permute(0, 2, 1)  # (B, HW, C)
            tokens.append(token)
            pos = build_2d_sincos_position_embedding(H, W, C, feat.device)
            pos_list.append(pos.unsqueeze(0).repeat(B, 1, 1))
        encoder_input = torch.cat(tokens, dim=1)
        encoder_pos = torch.cat(pos_list, dim=1)
        return encoder_input, encoder_pos


class TokenEncoder(nn.Module):
    def __init__(self, dim: int = CFG.DIM, num_layers: int = CFG.NUM_ENCODER_LAYERS,
                 num_heads: int = CFG.NUM_HEADS, ffn_dim: int = CFG.FFN_DIM, dropout: float = CFG.DROPOUT):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, src: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        return self.encoder(src + pos)


class DecoderLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.text_gate = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(inplace=True), nn.Linear(dim, dim))
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, text_vec: torch.Tensor) -> torch.Tensor:
        q = self.norm1(tgt + self.self_attn(tgt, tgt, tgt, need_weights=False)[0])
        k = memory
        v = memory
        cross = self.cross_attn(q, k, v, need_weights=False)[0]
        text_bias = self.text_gate(text_vec).unsqueeze(1)
        cross = self.norm2(q + cross + text_bias)
        out = self.norm3(cross + self.ffn(cross))
        return out


class ProposalDecoder(nn.Module):
    def __init__(self, dim: int = CFG.DIM, num_queries: int = CFG.NUM_QUERIES,
                 num_layers: int = CFG.NUM_DECODER_LAYERS, num_heads: int = CFG.NUM_HEADS,
                 dropout: float = CFG.DROPOUT):
        super().__init__()
        self.query_embed = nn.Parameter(torch.randn(num_queries, dim))
        self.layers = nn.ModuleList([
            DecoderLayer(dim, num_heads, dropout) for _ in range(num_layers)
        ])

    def forward(self, memory: torch.Tensor, text_vec: torch.Tensor) -> torch.Tensor:
        B = memory.size(0)
        queries = self.query_embed.unsqueeze(0).repeat(B, 1, 1)
        for layer in self.layers:
            queries = layer(queries, memory, text_vec)
        return queries


class PredictionHead(nn.Module):
    def __init__(self, dim: int = CFG.DIM):
        super().__init__()
        self.bbox = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, 4),
        )
        self.obj = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 2, 1),
        )

    def forward(self, query_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        boxes = torch.sigmoid(self.bbox(query_states))
        logits = self.obj(query_states).squeeze(-1)
        return {"pred_boxes": boxes, "pred_logits": logits}


class DETRLikeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.text = TextEncoder()
        self.image = ImageEncoder()
        self.encoder = TokenEncoder()
        self.decoder = ProposalDecoder()
        self.head = PredictionHead()

    def forward(self, images: torch.Tensor, tokens: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        tokens = {k: v.to(images.device) for k, v in tokens.items()}
        memory_in, pos = self.image(images)
        memory = self.encoder(memory_in, pos)
        text_vec = self.text(tokens)
        decoder_states = self.decoder(memory, text_vec)
        return self.head(decoder_states)


# ---------------------------------------------------------------------------
# Matching + losses
# ---------------------------------------------------------------------------


def box_cxcywh_to_xyxy(box: torch.Tensor) -> torch.Tensor:
    x_c, y_c, w, h = box.unbind(-1)
    half_w = w / 2.0
    half_h = h / 2.0
    return torch.stack((x_c - half_w, y_c - half_h, x_c + half_w, y_c + half_h), dim=-1)


def box_area(box: torch.Tensor) -> torch.Tensor:
    return (box[:, 2] - box[:, 0]).clamp(min=0) * (box[:, 3] - box[:, 1]).clamp(min=0)


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[:, 0])
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[:, 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[:, 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[:, 3])
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    union = area1[:, None] + area2 - inter_area
    iou = inter_area / union.clamp(min=1e-6)

    enclose_x1 = torch.min(boxes1[:, None, 0], boxes2[:, 0])
    enclose_y1 = torch.min(boxes1[:, None, 1], boxes2[:, 1])
    enclose_x2 = torch.max(boxes1[:, None, 2], boxes2[:, 2])
    enclose_y2 = torch.max(boxes1[:, None, 3], boxes2[:, 3])
    enclose_area = (enclose_x2 - enclose_x1).clamp(min=0) * (enclose_y2 - enclose_y1).clamp(min=0)
    return iou - (enclose_area - union) / enclose_area.clamp(min=1e-6)


def try_linear_sum_assignment(cost_matrix: torch.Tensor) -> Tuple[List[int], List[int]]:
    try:
        from scipy.optimize import linear_sum_assignment  # type: ignore

        row, col = linear_sum_assignment(cost_matrix.detach().cpu().numpy())
        return list(row), list(col)
    except Exception:
        rows: List[int] = []
        cols: List[int] = []
        remaining = list(range(cost_matrix.size(1)))
        for r in range(cost_matrix.size(0)):
            if not remaining:
                break
            best = min(remaining, key=lambda c: float(cost_matrix[r, c]))
            rows.append(r)
            cols.append(best)
            remaining.remove(best)
        return rows, cols


def match_predictions(outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]]) -> List[Tuple[int, torch.Tensor, torch.Tensor]]:
    pred_boxes = outputs["pred_boxes"]
    pred_logits = outputs["pred_logits"]
    device = pred_boxes.device
    matches: List[Tuple[int, torch.Tensor, torch.Tensor]] = []
    for b_idx, tgt in enumerate(targets):
        tgt_boxes = tgt["boxes"].to(device)
        if tgt_boxes.numel() == 0:
            continue
        cost_bbox = torch.cdist(pred_boxes[b_idx], tgt_boxes, p=1)
        cost_giou = 1.0 - generalized_box_iou(
            box_cxcywh_to_xyxy(pred_boxes[b_idx]),
            box_cxcywh_to_xyxy(tgt_boxes),
        )
        prob = pred_logits[b_idx].sigmoid().unsqueeze(1)
        cost_obj = -prob
        total_cost = (
            CFG.COST_BBOX * cost_bbox + CFG.COST_GIOU * cost_giou + CFG.COST_OBJ * cost_obj
        )
        row_ind, col_ind = try_linear_sum_assignment(total_cost)
        if not row_ind:
            continue
        matches.append(
            (
                b_idx,
                torch.as_tensor(row_ind, dtype=torch.long, device=device),
                torch.as_tensor(col_ind, dtype=torch.long, device=device),
            )
        )
    return matches


def detr_losses(outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    device = outputs["pred_boxes"].device
    matches = match_predictions(outputs, targets)
    obj_target = torch.zeros_like(outputs["pred_logits"], device=device)
    pred_matched: List[torch.Tensor] = []
    tgt_matched: List[torch.Tensor] = []
    for batch_idx, src_idx, tgt_idx in matches:
        obj_target[batch_idx, src_idx] = 1.0
        pred_matched.append(outputs["pred_boxes"][batch_idx, src_idx])
        tgt_matched.append(targets[batch_idx]["boxes"].to(device)[tgt_idx])

    if pred_matched:
        pred_boxes = torch.cat(pred_matched, dim=0)
        tgt_boxes = torch.cat(tgt_matched, dim=0)
        loss_bbox = F.l1_loss(pred_boxes, tgt_boxes)
        giou = generalized_box_iou(
            box_cxcywh_to_xyxy(pred_boxes),
            box_cxcywh_to_xyxy(tgt_boxes),
        )
        loss_giou = 1.0 - torch.diag(giou).mean()
    else:
        zero = outputs["pred_boxes"].sum() * 0.0
        loss_bbox = zero
        loss_giou = zero

    loss_obj = F.binary_cross_entropy_with_logits(outputs["pred_logits"], obj_target)
    return {
        "loss_bbox": loss_bbox * CFG.WEIGHT_BBOX,
        "loss_giou": loss_giou * CFG.WEIGHT_GIOU,
        "loss_obj": loss_obj * CFG.WEIGHT_OBJ,
    }


# ---------------------------------------------------------------------------
# Training & evaluation utilities
# ---------------------------------------------------------------------------


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                    device: torch.device, scaler: Optional[torch.amp.GradScaler] = None) -> Dict[str, float]:
    model.train()
    meter = {"loss": 0.0, "loss_bbox": 0.0, "loss_giou": 0.0, "loss_obj": 0.0}
    total = 0
    iterator = tqdm(loader, desc="train", leave=False, dynamic_ncols=True)
    for images, tokens, targets, _ in iterator:
        images = images.to(device)
        tokens = {k: v.to(device) for k, v in tokens.items()}
        optimizer.zero_grad(set_to_none=True)
        autocast_device = "cuda" if device.type == "cuda" else "cpu"
        with torch.amp.autocast(device_type=autocast_device, enabled=device.type == "cuda"):
            outputs = model(images, tokens)
            loss_dict = detr_losses(outputs, targets)
            loss = sum(loss_dict.values())
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        batch = images.size(0)
        total += batch
        meter["loss"] += float(loss.item()) * batch
        for k, v in loss_dict.items():
            meter[k] += float(v.item()) * batch
    return {k: v / max(1, total) for k, v in meter.items()}


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    ious: List[float] = []
    with torch.no_grad():
        iterator = tqdm(loader, desc="eval", leave=False, dynamic_ncols=True)
        for images, tokens, targets, _ in iterator:
            images = images.to(device)
            tokens = {k: v.to(device) for k, v in tokens.items()}
            outputs = model(images, tokens)
            preds = outputs["pred_boxes"].sigmoid() if outputs["pred_boxes"].dtype != torch.float32 else outputs["pred_boxes"]
            logits = outputs["pred_logits"].sigmoid()
            best_idx = logits.argmax(dim=1)
            chosen = preds[torch.arange(preds.size(0)), best_idx]
            for pred_box, tgt in zip(chosen, targets):
                boxes = tgt["boxes"].to(device)
                if boxes.numel() == 0:
                    continue
                pred_xyxy = box_cxcywh_to_xyxy(pred_box.unsqueeze(0))
                tgt_xyxy = box_cxcywh_to_xyxy(boxes)
                inter = generalized_box_iou(pred_xyxy, tgt_xyxy)
                ious.append(float(torch.diag(inter).mean().cpu()))
    return float(np.mean(ious)) if ious else 0.0


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, val_metric: float, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "val_metric": val_metric,
            "cfg": {k: getattr(CFG, k) for k in dir(CFG) if k.isupper()},
        },
        path,
    )


# ---------------------------------------------------------------------------
# CLI / pipeline
# ---------------------------------------------------------------------------


def build_dataloaders(tokenizer: AutoTokenizer, args: argparse.Namespace):
    train_pairs = resolve_dir_pairs(args.train_json_dir, args.train_jpg_dir, args.train_root)
    train_sets: List[Dataset] = []
    for json_dir, jpg_dir in train_pairs:
        json_files = find_json_files(json_dir)
        ds = QueryDataset(json_files, jpg_dir, resize_to=args.img_size, supervised_only=True)
        if len(ds) == 0:
            continue
        train_sets.append(ds)
    if not train_sets:
        raise RuntimeError("No training data found")
    train_ds = train_sets[0] if len(train_sets) == 1 else torch.utils.data.ConcatDataset(train_sets)
    collate = build_collate_fn(tokenizer)
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate,
        persistent_workers=args.num_workers > 0,
    )

    val_dl = None
    if args.val_root or (args.val_json_dir and args.val_jpg_dir):
        val_pairs = resolve_dir_pairs(args.val_json_dir, args.val_jpg_dir, args.val_root)
        val_sets: List[Dataset] = []
        for json_dir, jpg_dir in val_pairs:
            json_files = find_json_files(json_dir)
            ds = QueryDataset(json_files, jpg_dir, resize_to=args.img_size)
            if len(ds) == 0:
                continue
            val_sets.append(ds)
        if val_sets:
            val_ds = val_sets[0] if len(val_sets) == 1 else torch.utils.data.ConcatDataset(val_sets)
            val_dl = DataLoader(
                val_ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=torch.cuda.is_available(),
                collate_fn=collate,
                persistent_workers=args.num_workers > 0,
            )
    return train_dl, val_dl


def run_training(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(CFG.TEXT_MODEL)
    train_dl, val_dl = build_dataloaders(tokenizer, args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DETRLikeModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    best_val = -1.0
    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(model, train_dl, optimizer, device, scaler)
        msg = (
            f"Epoch {epoch}/{args.epochs} "
            f"loss={train_stats['loss']:.4f} bbox={train_stats['loss_bbox']:.4f} "
            f"giou={train_stats['loss_giou']:.4f} obj={train_stats['loss_obj']:.4f}"
        )
        print(msg)
        if val_dl is not None:
            val_iou = evaluate(model, val_dl, device)
            print(f"  -> val IoU: {val_iou:.4f}")
            if val_iou > best_val:
                best_val = val_iou
                save_checkpoint(model, optimizer, epoch, val_iou, args.ckpt_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lightweight DETR-like trainer")
    parser.add_argument("--train_root", default=CFG.TRAIN_ROOT)
    parser.add_argument("--val_root", default=CFG.VAL_ROOT)
    parser.add_argument("--train_json_dir", default=CFG.TRAIN_JSON_DIR)
    parser.add_argument("--train_jpg_dir", default=CFG.TRAIN_JPG_DIR)
    parser.add_argument("--val_json_dir", default=CFG.VAL_JSON_DIR)
    parser.add_argument("--val_jpg_dir", default=CFG.VAL_JPG_DIR)
    parser.add_argument("--img_size", type=int, default=CFG.IMG_SIZE)
    parser.add_argument("--batch_size", type=int, default=CFG.BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=CFG.EPOCHS)
    parser.add_argument("--lr", type=float, default=CFG.LR)
    parser.add_argument("--num_workers", type=int, default=CFG.NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt_path", type=str, default=CFG.CKPT_PATH)
    return parser.parse_args()


def main():
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
