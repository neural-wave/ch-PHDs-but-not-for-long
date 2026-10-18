The original fitted champion binary was not available in this checkout.

From the package root, regenerate the fixed final recipe with:

```sh
python src/train_ensemble.py --mode final --device cpu --output model_checkpoints/final
```

Use `--device cuda` with an appropriate CUDA PyTorch installation. Training
creates `model.pt` and matching provenance/calibration records here. This full
fit is separate from the isolated one-epoch trial; no trial artifact is copied
here. Consult the package README before inference or jury handoff.
