import torch
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video

from nunchaku.models.transformers.transformer_wan import NunchakuWanTransformer3DModel
from nunchaku.utils import get_gpu_memory, get_precision

model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

# Load the quantized transformer (int4 on Turing/Ampere/Ada, fp4 on Blackwell)
transformer = NunchakuWanTransformer3DModel.from_pretrained(
    f"nunchaku-tech/nunchaku-wan2.1/wan2.1-t2v-1.3b-svdq-{get_precision()}.safetensors"
)

vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipe = WanPipeline.from_pretrained(model_id, transformer=transformer, vae=vae, torch_dtype=torch.bfloat16)

if get_gpu_memory() > 18:
    pipe.enable_model_cpu_offload()
else:
    # per-block offloading for low VRAM; this is also how the 14B model fits on 24 GB
    transformer.set_offload(True, num_blocks_on_gpu=1)
    pipe._exclude_from_cpu_offload.append("transformer")
    pipe.enable_sequential_cpu_offload()

prompt = "A cat walks on the grass, realistic style."
negative_prompt = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, "
    "overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, "
    "poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
    "still picture, messy background, three legs, many people in the background, walking backwards"
)

output = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=480,
    width=832,
    num_frames=81,
    guidance_scale=6.0,
    num_inference_steps=50,
    generator=torch.Generator().manual_seed(0),
).frames[0]
export_to_video(output, "wan2.1-t2v-1.3b.mp4", fps=16)
