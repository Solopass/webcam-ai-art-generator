"""Isolate WHERE the pipeline produces garbage, and what it costs.

Runs each configuration in its own subprocess (so a crash or a VRAM leak in one
cannot contaminate the next), feeds a fixed synthetic portrait through the real
TensorRT engine, and reports per-stage finiteness plus timings.

    venv\\Scripts\\python.exe diagnose.py

Everything lands in logs/diagnose.log.
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, "logs")

CONFIGS = [
    # (cfg_type, guidance, cuda_graph)  -- 'none' is the known-good baseline
    ("none", 1.0, False),
    ("self", 1.6, False),
    ("self", 1.2, False),
    ("none", 1.0, True),
]

ITERS = 30


def test_image():
    """Deterministic 512x512 pseudo-portrait: gradient wall + skin-tone head."""
    import cv2
    import numpy as np
    img = np.zeros((512, 512, 3), np.uint8)
    grad = np.linspace(40, 150, 512, dtype=np.uint8)
    img[:] = np.repeat(grad[None, :], 512, axis=0)[:, :, None]
    cv2.ellipse(img, (256, 300), (150, 190), 0, 0, 360, (150, 175, 205), -1)
    cv2.ellipse(img, (256, 250), (100, 125), 0, 0, 360, (165, 190, 220), -1)
    cv2.circle(img, (215, 235), 14, (60, 55, 50), -1)
    cv2.circle(img, (297, 235), 14, (60, 55, 50), -1)
    cv2.ellipse(img, (256, 305), (40, 18), 0, 0, 180, (80, 70, 90), -1)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def run_child(args):
    import numpy as np
    import torch
    from PIL import Image
    from diffusers import AutoencoderTiny, StableDiffusionPipeline
    from streamdiffusion import StreamDiffusion
    from streamdiffusion.acceleration.tensorrt import accelerate_with_tensorrt
    from streamdiffusion.image_utils import postprocess_image

    def out(msg):
        print(msg, flush=True)

    out(f"### cfg_type={args.cfg_type} guidance={args.guidance} "
        f"cuda_graph={args.cuda_graph} t_index={args.t_index}")

    pipe = StableDiffusionPipeline.from_pretrained(
        "KBlueLeaf/kohaku-v2.1", torch_dtype=torch.float16, safety_checker=None).to("cuda")
    pipe.vae = AutoencoderTiny.from_pretrained(
        "madebyollin/taesd", torch_dtype=torch.float16).to("cuda")

    stream = StreamDiffusion(
        pipe, t_index_list=[args.t_index], torch_dtype=torch.float16,
        cfg_type=args.cfg_type, do_add_noise=True, use_denoising_batch=True,
        frame_buffer_size=1)
    stream.load_lcm_lora()
    stream.fuse_lora()
    stream.prepare(prompt="1girl, masterpiece, anime key visual",
                   negative_prompt="blurry, deformed",
                   num_inference_steps=50,
                   guidance_scale=args.guidance, delta=1.0)
    stream = accelerate_with_tensorrt(
        stream, os.path.join(HERE, "engines_tinyvae_fb1"),
        max_batch_size=1, use_cuda_graph=args.cuda_graph,
        engine_build_options={"opt_batch_size": 1})

    # Trace finiteness at each stage inside the denoise step.
    trace = {}
    original_unet_step = stream.unet_step

    def traced_unet_step(x_t_latent, t_list, idx=None):
        x0, model_pred = original_unet_step(x_t_latent, t_list, idx)
        trace["latent_in"] = bool(torch.isfinite(x_t_latent).all())
        trace["model_pred"] = bool(torch.isfinite(model_pred).all())
        trace["x0"] = bool(torch.isfinite(x0).all())
        trace["latent_absmax"] = float(x_t_latent.abs().max())
        trace["pred_absmax"] = float(model_pred.float().abs().max())
        return x0, model_pred

    stream.unet_step = traced_unet_step

    img = Image.fromarray(test_image())
    first_bad = None
    bad_count = 0
    times = []

    for i in range(ITERS):
        t0 = time.perf_counter()
        result = stream(img)
        
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

        arr = postprocess_image(result, output_type="np")[0]
        finite = np.isfinite(arr)
        n_bad = int((~finite).sum())
        stock = getattr(stream, "stock_noise", None)
        stock_max = float(stock.float().abs().max()) if stock is not None else float("nan")

        if n_bad:
            bad_count += 1
            if first_bad is None:
                first_bad = i
        if i < 3 or n_bad or i == ITERS - 1:
            rng = (f"[{arr[finite].min():.3f}, {arr[finite].max():.3f}]"
                   if finite.any() else "ALL NON-FINITE")
            out(f"  iter {i:>2}: bad={n_bad:>7} range={rng} "
                f"latent_in={trace.get('latent_in')} model_pred={trace.get('model_pred')} "
                f"x0={trace.get('x0')} latent|max|={trace.get('latent_absmax'):.1f} "
                f"pred|max|={trace.get('pred_absmax'):.1f} stock|max|={stock_max:.1f}")

    warm = times[5:] or times
    out(f"  RESULT: {bad_count}/{ITERS} frames non-finite"
        + (f" (first at iter {first_bad})" if first_bad is not None else "")
        + f" | {sum(warm)/len(warm)*1000:.1f} ms/iter "
          f"({len(warm)/sum(warm):.1f} FPS ceiling)")
    return 0


def run_parent():
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, "diagnose.log")
    venv = os.path.join(HERE, "venv", "Scripts", "python.exe")
    python = venv if os.path.exists(venv) else sys.executable

    with open(log_path, "w", encoding="utf-8", buffering=1) as log:
        def emit(line):
            print(line, flush=True)
            log.write(line + "\n")

        emit(f"diagnose.py — {time.strftime('%Y-%m-%d %H:%M:%S')}")
        emit(f"{len(CONFIGS)} configurations x {ITERS} iterations\n")

        for cfg_type, guidance, cuda_graph in CONFIGS:
            cmd = [python, "-u", os.path.abspath(__file__), "--_child",
                   "--cfg_type", cfg_type, "--guidance", str(guidance)]
            if cuda_graph:
                cmd.append("--cuda_graph")
            proc = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True,
                                  encoding="utf-8", errors="replace")
            for line in (proc.stdout or "").splitlines():
                if line.startswith("###") or line.startswith("  "):
                    emit(line)
            if proc.returncode != 0:
                emit(f"  CRASHED (exit {proc.returncode}):")
                for line in (proc.stderr or "").strip().splitlines()[-25:]:
                    emit("    " + line)
            emit("")

        emit(f"Written to {log_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--_child", action="store_true")
    p.add_argument("--cfg_type", default="none")
    p.add_argument("--guidance", type=float, default=1.0)
    p.add_argument("--cuda_graph", action="store_true")
    p.add_argument("--t_index", type=int, default=27)
    a = p.parse_args()
    sys.exit(run_child(a) if a._child else run_parent())
