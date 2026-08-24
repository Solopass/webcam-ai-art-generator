import torch
from diffusers import UNet2DConditionModel, ControlNetModel

class UNetWithControlNet(torch.nn.Module):
    def __init__(self, unet, controlnet):
        super().__init__()
        self.unet = unet
        self.controlnet = controlnet
    
    def forward(self, sample, timestep, encoder_hidden_states, control_image):
        down, mid = self.controlnet(
            sample, timestep, encoder_hidden_states, controlnet_cond=control_image, return_dict=False
        )
        return self.unet(
            sample, timestep, encoder_hidden_states,
            down_block_additional_residuals=down,
            mid_block_additional_residual=mid,
            return_dict=False
        )[0]

print("Module defined successfully.")
