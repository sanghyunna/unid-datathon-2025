# unid-datathon-2025

## Layout-aware VLM pipeline

`sanghyunna/baseline_final.py` rewrites the original baseline with the following upgrades:

- Upgraded the Vision backbone to ResNet50
- Upgraded the GRU dimension to 1024 for expressiveness


### Install runtime dependencies

```bash
pip install -r requirements.txt
```

### Typical usage

```
cd sanghyunna
python baseline_final.py fit
python baseline_final.py predict --ckpt ./outputs_baseline_best/ckpt/cross_attn_vlm_ep19.0.pth --out_csv ./outputs_baseline_best/preds/test_pred_ep19.csv
```