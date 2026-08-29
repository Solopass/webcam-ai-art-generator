# StreamDiffusion patches

This project depends on modifications to the vendored StreamDiffusion checkout.
`StreamDiffusion/` is gitignored, so these patches are the only copy that lives
in version control. **Without them a fresh clone cannot build a TensorRT
engine** — it fails with:

```
TypeError: compile_unet() got an unexpected keyword argument 'timing_cache'
```

Base commit: `b623251` of https://github.com/cumulo-autumn/StreamDiffusion.git

`setup.ps1` applies these automatically after cloning. To do it by hand:

```
cd StreamDiffusion
git checkout b623251
git am --3way ../patches/*.patch
```

## What they change

- `acceleration/tensorrt/__init__.py`, `builder.py`, `utilities.py` — `**kwargs`
  passthrough on the `compile_*` helpers, `timing_cache` plumbed through to the
  builder, and a TensorRT API fix (`get_bindings_per_profile` was removed in
  newer TensorRT).
- `acceleration/tensorrt/engine.py`, `pipeline.py`, `utils/wrapper.py` — native
  ControlNet conditioning support.

## Regenerating

After committing further changes inside `StreamDiffusion/`:

```
cd StreamDiffusion
git format-patch b623251..HEAD --stdout > ../patches/0001-streamdiffusion-controlnet-and-trt.patch
```
