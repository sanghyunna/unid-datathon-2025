# unid-datathon-2025

## Layout-aware VLM pipeline

`sanghyunna/new_model.py` rewrites the original baseline with the following upgrades:

- DocLayout YOLO serves as the vision backbone (multi-scale features captured before the detection head, with automatic TinyCNN fallback when the weights are missing).
- DocLayout YOLO serves as the vision backbone. We now pull the official checkpoints from Hugging Face (`juliozhao/DocLayout-YOLO-DocStructBench`) using `--vision_repo_id` / `--vision_filename`, or you can point `--vision_weights` at your own `.pt` file.
- The Korean-friendly `BAAI/bge-m3` transformer replaces the legacy GRU text encoder; prompts can be prepended to each instruction for better grounding.
- A stack of cross-attention blocks fuses the modalities before regressing bounding boxes, and the training loop now tracks mid-epoch mIoU for early feedback.

### Install runtime dependencies

```bash
pip install -r sanghyunna/requirements_new_model.txt
```

### Typical usage

```bash
# train only
python sanghyunna/new_model.py train \
	--train_root ../data/train_valid/train \
	--vision_repo_id juliozhao/DocLayout-YOLO-DocStructBench \
	--vision_filename doclayout_yolo_docstructbench_imgsz1024.pt \
	--batch_size 8 --epochs 5 --save_ckpt outputs/ckpt/layout_regressor.pth

# one-shot fit (train + val eval csv + test prediction csv)
python sanghyunna/new_model.py fit \
	--train_root ../data/train_valid/train \
	--val_root ../data/train_valid/valid \
	--test_root ../data/open/test \
	--epochs 6 --lr 2e-4 --save_ckpt outputs/ckpt/layout_regressor.pth

# evaluation or test-time prediction
python sanghyunna/new_model.py eval --ckpt outputs/ckpt/layout_regressor.pth --val_root ../data/train_valid/valid
python sanghyunna/new_model.py predict --ckpt outputs/ckpt/layout_regressor.pth --test_root ../data/open/test
```

Key flags:

- `--text_prompt` lets you prepend Korean instructions without touching the raw query JSON.
- `--vision_feature_level` controls which DocLayout YOLO scale (default: highest resolution) feeds the cross-attention stack.
- `--freeze_text` / `--freeze_vision` decide whether to fine-tune the heavy backbones.

Checkpoint metadata stores every modeling hyperparameter (encoders, prompt, text length, feature level, etc.), so `--resume_ckpt` automatically restores matching tokenizers and image sizing.
