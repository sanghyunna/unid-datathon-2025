import os
import io
import json
import math
import time
import random
import argparse
import zipfile
from glob import glob
from typing import List, Tuple, Dict, Any

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# torchvision 백본 사용 가능 여부 체크
try:
    from torchvision.models import resnet18, ResNet18_Weights
    _BACKBONE_OK = True
except Exception:
    _BACKBONE_OK = False


class CFG:
    # Core
    IMG_SIZE: int = 512
    EPOCHS: int = 10
    LEARNING_RATE: float = 1e-4
    BATCH_SIZE: int = 8
    SEED: int = 42
    DIM: int = 256
    NUM_WORKERS: int = 2
    NO_PRETRAIN: bool = False  # True → disable ImageNet weights

    # Paths (override by CLI if desired)
    JSON_DIR: str = "./data/json"
    JPG_DIR: str = None
    CKPT_PATH: str = "./outputs/ckpt/cross_attn_vlm.pth"
    EVAL_CSV: str = "./outputs/preds/eval_pred.csv"
    PRED_CSV: str = "./outputs/preds/test_pred.csv"
    SUBMISSION_ZIP: str = "./outputs/submission.zip"

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#Data Pre-processing (helpers)
def find_jsons(json_dir: str) -> List[str]:
    if os.path.isdir(json_dir):
        return sorted(glob(os.path.join(json_dir, "*.json")))
    raise FileNotFoundError(f"json_dir not found: {json_dir}")


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_image_path(json_path: str, data: Dict[str, Any], jpg_dir: str = None) -> str:
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

#CustomDataset
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
    def __init__(self, json_files: List[str], jpg_dir: str = None, vocab: Vocab = None,
                 build_vocab: bool = False, resize_to: Tuple[int, int] = (CFG.IMG_SIZE, CFG.IMG_SIZE)):
        self.items = []
        for jf in json_files:
            data = read_json(jf)
            ann = data.get("learning_data_info", {}).get("annotation", [])
            img_path = get_image_path(jf, data, jpg_dir=jpg_dir)
            for a in ann:
                if not is_visual_ann(a):
                    continue
                qid = a.get("instance_id", "")
                qtxt = str(a.get("visual_instruction", "")).strip()
                bbox = a.get("bounding_box", None)  # train/val has bbox; test may be None
                cname = a.get("class_name", "")
                self.items.append({
                    "json": jf, "img": img_path,
                    "query_id": qid, "query": qtxt,
                    "bbox": bbox, "class_name": cname,
                })

        self.vocab = vocab if vocab is not None else Vocab(min_freq=1)
        if build_vocab:
            self.vocab.build([it["query"] for it in self.items])

        self.resize_to = resize_to
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

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        it = self.items[idx]
        img = Image.open(it["img"]).convert("RGB")
        W, H = img.size
        if self.tf is not None:
            img_t = self.tf(img)
        else:
            img = img.resize(self.resize_to, Image.BILINEAR)
            img_t = self._pil_to_tensor(img)

        ids = self.vocab.encode(it["query"], max_len=40)
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
    return imgs, ids, lens, targets, meta

#model define
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
                weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
                m = resnet18(weights=weights)
                layers = list(m.children())[:-2]  # (B, 512, H/32, W/32)
                self.backbone = nn.Sequential(*layers)
                self.proj = nn.Conv2d(512, out_dim, 1)
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

#Train
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


def make_loader(json_dir: str, jpg_dir: str, vocab: Vocab = None, build_vocab: bool = False,
                batch_size: int = CFG.BATCH_SIZE, img_size: int = CFG.IMG_SIZE,
                num_workers: int = CFG.NUM_WORKERS, shuffle: bool = False):
    json_files = find_jsons(json_dir)
    ds = UniDSet(json_files, jpg_dir=jpg_dir, vocab=vocab, build_vocab=build_vocab,
                 resize_to=(img_size, img_size))
    if build_vocab:
        # use only supervised samples (have bbox)
        sup_idx = [i for i in range(len(ds)) if ds[i]["target"] is not None]
        if len(sup_idx) == 0:
            raise RuntimeError("No supervised samples (no bboxes) in given json_dir.")
        ds = torch.utils.data.Subset(ds, sup_idx)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                    num_workers=num_workers, collate_fn=collate_fn)
    return ds, dl


def train_loop(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Build dataset + vocab on train jsons
    train_ds, train_dl = make_loader(args.json_dir, args.jpg_dir, vocab=None, build_vocab=True,
                                     batch_size=args.batch_size, img_size=args.img_size,
                                     num_workers=args.num_workers, shuffle=True)

    model = CrossAttnVLM(vocab_size=len(train_ds.dataset.vocab.itos) if isinstance(train_ds, torch.utils.data.Subset) else len(train_ds.vocab.itos),
                         dim=args.dim, pretrained_backbone=not args.no_pretrain, img_size=args.img_size).to(device)

    # Resolve vocab (Subset wrapper case)
    vocab = train_ds.dataset.vocab if isinstance(train_ds, torch.utils.data.Subset) else train_ds.vocab

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    total_samples = len(train_ds)
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for imgs, ids, lens, targets, meta in train_dl:
            imgs = imgs.to(device); ids = ids.to(device); lens = lens.to(device)
            t = torch.stack([tar for tar in targets if tar is not None], dim=0).to(device)  # (B,4)
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

    os.makedirs(os.path.dirname(args.save_ckpt), exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "vocab_itos": vocab.itos,
        "dim": args.dim,
        "no_pretrain": args.no_pretrain,
        "img_size": args.img_size,
    }, args.save_ckpt)
    print(f"[Saved] {args.save_ckpt}")

#Inference
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

    json_files = find_jsons(args.json_dir)
    ds = UniDSet(json_files, jpg_dir=args.jpg_dir, vocab=vocab, build_vocab=False,
                 resize_to=(img_size, img_size))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, collate_fn=collate_fn)

    rows = []; ious = []
    with torch.no_grad():
        for imgs, ids, lens, targets, meta in dl:
            imgs = imgs.to(device); ids = ids.to(device); lens = lens.to(device)
            pred = model(imgs, ids, lens)  # (B,4) normalized
            for i in range(imgs.size(0)):
                W, H = meta[i]["orig_size"]
                cx, cy, nw, nh = [float(v) for v in pred[i].cpu().numpy().tolist()]
                x = (cx - nw / 2.0) * W; y = (cy - nh / 2.0) * H
                w = nw * W; h = nh * H
                rows.append({
                    "query_id": meta[i]["query_id"], "query_text": meta[i]["query_text"],
                    "pred_x": x, "pred_y": y, "pred_w": w, "pred_h": h
                })
                if targets[i] is not None:
                    gt = [float(v) for v in targets[i].numpy().tolist()]
                    gx = (gt[0] - gt[2] / 2.0) * W; gy = (gt[1] - gt[3] / 2.0) * H
                    gw = gt[2] * W; gh = gt[3] * H
                    ious.append(iou_xywh_pixel([x, y, w, h], [gx, gy, gw, gh]))

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    import pandas as pd
    df = pd.DataFrame(rows, columns=["query_id", "query_text", "pred_x", "pred_y", "pred_w", "pred_h"])
    df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    print(f"[Saved] {args.out_csv}")
    if ious:
        print(f"[Eval] mIoU={float(np.mean(ious))}")
    else:
        print("[Eval] No GT found; mIoU not computed.")


def predict_loop(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab, img_size = _load_model_from_ckpt(args.ckpt, device)

    json_files = find_jsons(args.json_dir)
    ds = UniDSet(json_files, jpg_dir=args.jpg_dir, vocab=vocab, build_vocab=False,
                 resize_to=(img_size, img_size))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, collate_fn=collate_fn)

    rows = []
    with torch.no_grad():
        for imgs, ids, lens, targets, meta in dl:
            imgs = imgs.to(device); ids = ids.to(device); lens = lens.to(device)
            pred = model(imgs, ids, lens)  # (B,4) normalized
            for i in range(imgs.size(0)):
                W, H = meta[i]["orig_size"]
                cx, cy, nw, nh = [float(v) for v in pred[i].cpu().numpy().tolist()]
                x = (cx - nw / 2.0) * W; y = (cy - nh / 2.0) * H
                w = nw * W; h = nh * H
                rows.append({
                    "query_id": meta[i]["query_id"], "query_text": meta[i]["query_text"],
                    "pred_x": x, "pred_y": y, "pred_w": w, "pred_h": h
                })

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    import pandas as pd
    df = pd.DataFrame(rows, columns=["query_id", "query_text", "pred_x", "pred_y", "pred_w", "pred_h"])
    df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    print(f"[Saved] {args.out_csv}")

    #Submission
    def zip_submission(csv_path: str, zip_path: str):
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        arcname = os.path.basename(csv_path)
        zf.write(csv_path, arcname=arcname)
    print(f"[Submission] Zipped {csv_path} → {zip_path}")

    #CLI
    def get_args():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_common(p):
        p.add_argument("--json_dir", type=str, default=CFG.JSON_DIR, help="Directory with JSON files")
        p.add_argument("--jpg_dir", type=str, default=CFG.JPG_DIR, help="Directory with JPG images")
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

    # eval
    p_eval = sub.add_parser("eval")
    add_common(p_eval)
    p_eval.add_argument("--ckpt", type=str, required=True)
    p_eval.add_argument("--out_csv", type=str, default=CFG.EVAL_CSV)

    # predict
    p_pred = sub.add_parser("predict")
    add_common(p_pred)
    p_pred.add_argument("--ckpt", type=str, required=True)
    p_pred.add_argument("--out_csv", type=str, default=CFG.PRED_CSV)

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