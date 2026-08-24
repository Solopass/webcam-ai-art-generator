import torch
from diffusers import StableDiffusionPipeline
from streamdiffusion import StreamDiffusion

print("Loading pipe...")
pipe = StableDiffusionPipeline.from_pretrained("KBlueLeaf/kohaku-v2.1").to(
    device=torch.device("cuda"),
    dtype=torch.float16,
)

print("Initializing StreamDiffusion...")
try:
    stream = StreamDiffusion(
        pipe,
        t_index_list=[32],
        torch_dtype=torch.float16,
        cfg_type="self",
        do_add_noise=True,
    )
    print("SUCCESS")
except Exception as e:
    print(f"FAILED: {e}")
