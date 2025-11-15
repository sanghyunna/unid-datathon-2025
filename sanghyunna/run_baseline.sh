python sanghyunna/baseline.py eval \
  --json_dir data/train_valid/valid/press_json \
  --jpg_dir data/train_valid/valid/press_jpg \
  --batch_size 8 \
  --dim 256 \
  --num_workers 4 \
  --ckpt outputs/ckpt/cross_attn_vlm_press.pth \
  --out_csv outputs/preds/press_eval.csv