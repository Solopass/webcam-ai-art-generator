import os
import gc
import torch
import onnx
import argparse
from diffusers import ControlNetModel, StableDiffusionPipeline

from streamdiffusion.acceleration.tensorrt.models import BaseModel
from streamdiffusion.acceleration.tensorrt.utilities import build_engine

class UNetControlNetWrapper(torch.nn.Module):
    def __init__(self, unet, controlnet):
        super().__init__()
        self.unet = unet
        self.controlnet = controlnet

    def forward(self, sample, timestep, encoder_hidden_states, controlnet_cond):
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=controlnet_cond,
            return_dict=False,
        )
        return self.unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
            return_dict=False,
        )[0]

class UNetMultiControlNetWrapper(torch.nn.Module):
    def __init__(self, unet, controlnets):
        super().__init__()
        self.unet = unet
        from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
        self.controlnet = MultiControlNetModel(controlnets)

    def forward(self, sample, timestep, encoder_hidden_states, controlnet_cond):
        # Split [B, 6, H, W] into two [B, 3, H, W]
        cond1 = controlnet_cond[:, 0:3, :, :]
        cond2 = controlnet_cond[:, 3:6, :, :]
        
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=[cond1, cond2],
            conditioning_scale=[1.0, 1.0],
            return_dict=False,
        )
        return self.unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
            return_dict=False,
        )[0]


class UNetControlNetData(BaseModel):
    def __init__(self, fp16=True, device="cuda", max_batch_size=16, min_batch_size=1, embedding_dim=768, text_maxlen=77, unet_dim=4, cond_channels=3):
        super().__init__(fp16=fp16, device=device, max_batch_size=max_batch_size, min_batch_size=min_batch_size, embedding_dim=embedding_dim, text_maxlen=text_maxlen)
        self.unet_dim = unet_dim
        self.name = "UNetControlNet"
        self.cond_channels = cond_channels

    def get_input_names(self):
        return ["sample", "timestep", "encoder_hidden_states", "controlnet_cond"]

    def get_output_names(self):
        return ["latent"]

    def get_dynamic_axes(self):
        return {
            "sample": {0: "2B", 2: "H", 3: "W"},
            "timestep": {0: "2B"},
            "encoder_hidden_states": {0: "2B"},
            "controlnet_cond": {0: "2B", 2: "8H", 3: "8W"},
            "latent": {0: "2B", 2: "H", 3: "W"},
        }

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        (min_batch, max_batch, min_image_height, max_image_height, min_image_width, max_image_width, min_latent_height, max_latent_height, min_latent_width, max_latent_width) = self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)
        return {
            "sample": [(min_batch, self.unet_dim, min_latent_height, min_latent_width), (batch_size, self.unet_dim, latent_height, latent_width), (max_batch, self.unet_dim, max_latent_height, max_latent_width)],
            "timestep": [(min_batch,), (batch_size,), (max_batch,)],
            "encoder_hidden_states": [(min_batch, self.text_maxlen, self.embedding_dim), (batch_size, self.text_maxlen, self.embedding_dim), (max_batch, self.text_maxlen, self.embedding_dim)],
            "controlnet_cond": [(min_batch, self.cond_channels, min_image_height, min_image_width), (batch_size, self.cond_channels, image_height, image_width), (max_batch, self.cond_channels, max_image_height, max_image_width)],
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            "sample": (2 * batch_size, self.unet_dim, latent_height, latent_width),
            "timestep": (2 * batch_size,),
            "encoder_hidden_states": (2 * batch_size, self.text_maxlen, self.embedding_dim),
            "controlnet_cond": (2 * batch_size, self.cond_channels, image_height, image_width),
            "latent": (2 * batch_size, 4, latent_height, latent_width),
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        dtype = torch.float16 if self.fp16 else torch.float32
        return (
            torch.randn(2 * batch_size, self.unet_dim, latent_height, latent_width, dtype=torch.float32, device=self.device),
            torch.ones((2 * batch_size,), dtype=torch.float32, device=self.device),
            torch.randn(2 * batch_size, self.text_maxlen, self.embedding_dim, dtype=dtype, device=self.device),
            torch.randn(2 * batch_size, self.cond_channels, image_height, image_width, dtype=dtype, device=self.device),
        )

MODELS = {
    "depth": "lllyasviel/control_v11f1p_sd15_depth",
    "canny": "lllyasviel/control_v11p_sd15_canny",
    "lineart": "lllyasviel/control_v11p_sd15_lineart",
    "openpose": "lllyasviel/control_v11p_sd15_openpose",
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, default="depth", choices=["depth", "canny", "lineart", "openpose", "multi"])
    parser.add_argument("--batch", type=int, default=8, help="Max batch size")
    args = parser.parse_args()

    print("Loading Kohaku UNet...")
    pipe = StableDiffusionPipeline.from_pretrained("KBlueLeaf/kohaku-v2.1", torch_dtype=torch.float16).to("cuda")
    
    cond_channels = 3
    if args.type == "multi":
        print("Loading Multi-ControlNet (Depth + Canny)...")
        c1 = ControlNetModel.from_pretrained(MODELS["depth"], torch_dtype=torch.float16).to("cuda")
        c2 = ControlNetModel.from_pretrained(MODELS["canny"], torch_dtype=torch.float16).to("cuda")
        wrapper = UNetMultiControlNetWrapper(pipe.unet, [c1, c2])
        cond_channels = 6
    else:
        print(f"Loading ControlNet {args.type}...")
        controlnet = ControlNetModel.from_pretrained(MODELS[args.type], torch_dtype=torch.float16).to("cuda")
        wrapper = UNetControlNetWrapper(pipe.unet, controlnet)

    wrapper.to("cuda").eval()
    
    model_data = UNetControlNetData(fp16=True, max_batch_size=args.batch, min_batch_size=1, cond_channels=cond_channels)
    
    os.makedirs("engines_controlnet", exist_ok=True)
    onnx_path = f"engines_controlnet/unet_{args.type}.onnx"
    engine_path = f"engines_controlnet/unet_{args.type}.engine"
    
    print(f"Exporting ONNX to {onnx_path}...")
    with torch.inference_mode(), torch.autocast("cuda"):
        inputs = model_data.get_sample_input(args.batch, 512, 512)
        torch.onnx.export(
            wrapper,
            inputs,
            onnx_path,
            export_params=True,
            opset_version=16,
            do_constant_folding=True,
            input_names=model_data.get_input_names(),
            output_names=model_data.get_output_names(),
            dynamic_axes=model_data.get_dynamic_axes(),
        )
    
    # Cap workspace to 8GB to prevent 3080 Ti from crashing during massive multi-controlnet build
    print("Building TRT Engine (This will take 10-15 minutes)...")
    build_engine(
        engine_path, 
        onnx_path, 
        model_data, 
        opt_image_height=512, 
        opt_image_width=512, 
        opt_batch_size=args.batch, 
        build_static_batch=False, 
        build_dynamic_shape=False, 
        build_all_tactics=False, 
        timing_cache="trt_global_timing.cache",
        build_enable_refit=False,
    )
    
    print(f"Done compiling ControlNet {args.type}!")

if __name__ == "__main__":
    main()
