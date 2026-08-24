import os
import gc
import torch
import onnx
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

class UNetControlNetData(BaseModel):
    def __init__(self, fp16=True, device="cuda", max_batch_size=16, min_batch_size=1, embedding_dim=768, text_maxlen=77, unet_dim=4):
        super().__init__(fp16=fp16, device=device, max_batch_size=max_batch_size, min_batch_size=min_batch_size, embedding_dim=embedding_dim, text_maxlen=text_maxlen)
        self.unet_dim = unet_dim
        self.name = "UNetControlNet"

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
            "controlnet_cond": [(min_batch, 3, min_image_height, min_image_width), (batch_size, 3, image_height, image_width), (max_batch, 3, max_image_height, max_image_width)],
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            "sample": (2 * batch_size, self.unet_dim, latent_height, latent_width),
            "timestep": (2 * batch_size,),
            "encoder_hidden_states": (2 * batch_size, self.text_maxlen, self.embedding_dim),
            "controlnet_cond": (2 * batch_size, 3, image_height, image_width),
            "latent": (2 * batch_size, 4, latent_height, latent_width),
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        dtype = torch.float16 if self.fp16 else torch.float32
        return (
            torch.randn(2 * batch_size, self.unet_dim, latent_height, latent_width, dtype=torch.float32, device=self.device),
            torch.ones((2 * batch_size,), dtype=torch.float32, device=self.device),
            torch.randn(2 * batch_size, self.text_maxlen, self.embedding_dim, dtype=dtype, device=self.device),
            torch.randn(2 * batch_size, 3, image_height, image_width, dtype=dtype, device=self.device),
        )

def main():
    print("Loading Kohaku...")
    pipe = StableDiffusionPipeline.from_pretrained("KBlueLeaf/kohaku-v2.1", torch_dtype=torch.float16).to("cuda")
    print("Loading ControlNet Depth...")
    controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float16).to("cuda")
    
    wrapper = UNetControlNetWrapper(pipe.unet, controlnet)
    wrapper.to("cuda").eval()
    
    # Use standard 4 max batch size (for ultra low latency 1-frame batch fb1 + cfg + u/c)
    model_data = UNetControlNetData(fp16=True, max_batch_size=4, min_batch_size=1)
    
    os.makedirs("engines_controlnet", exist_ok=True)
    onnx_path = "engines_controlnet/unet_depth.onnx"
    engine_path = "engines_controlnet/unet_depth.engine"
    
    print("Exporting ONNX (with external data to bypass 2GB limit)...")
    with torch.inference_mode(), torch.autocast("cuda"):
        inputs = model_data.get_sample_input(4, 512, 512)
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
    
    # We skip ONNX GraphSurgeon completely. TRT will optimize the raw graph.
    print("Building TRT Engine (This will take 10-15 minutes)...")
    build_engine(engine_path, onnx_path, model_data, 512, 512, 4, True, False, False, None, "trt_global_timing.cache")
    
    print("Done compiling ControlNet!")

if __name__ == "__main__":
    main()
