import os
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from addit.addit_flux_pipeline import AdditFluxPipeline
from addit.addit_flux_transformer import AdditFluxTransformer2DModel
from addit.addit_scheduler import AdditFlowMatchEulerDiscreteScheduler
from addit.addit_methods import add_object_real
from CatVTON.model.pipeline import CatVTONPipeline
from CatVTON.utils import init_weight_dtype, resize_and_crop, resize_and_padding
from torch.utils.data import Dataset, DataLoader
from diffusers import AutoPipelineForInpainting, DiffusionPipeline, FluxTransformer2DModel
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils import load_image
from ObjectClear.objectclear.pipelines import ObjectClearPipeline
from ObjectClear.objectclear.utils import resize_by_short_side
from huggingface_hub import snapshot_download
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_closing
from ip_adapter import IPAdapterXL
from ultralytics import YOLOWorld
from dotenv import load_dotenv
from openai import OpenAI
from io import BytesIO
from groq import Groq
import numpy as np
import torch
import base64
import torch
import re
import json
import os
import random
from openai import OpenAI

# Load environment variables
load_dotenv(".env")

# Define the Device
resize_size = 840
device_1 = "cuda:0" if torch.cuda.is_available() else "cpu"

# Vision Reasoner 7B
vision_reasoner_path="Ricky06662/VisionReasoner-7B"
vision_reasoner = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        vision_reasoner_path,
                        torch_dtype=torch.float16,
                        device_map=device_1).eval()
vision_reasoner_processor = AutoProcessor.from_pretrained(
                        vision_reasoner_path, 
                        padding_side="left")
# SAM-2 by Meta
segmentation_model_path ="facebook/sam2-hiera-large"
segmentation_model = SAM2ImagePredictor.from_pretrained(
                        segmentation_model_path)
# SDXL
diffusion_model_path="diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
diffusion_model = AutoPipelineForInpainting.from_pretrained(
                        diffusion_model_path, 
                        torch_dtype=torch.float16,
                        variant="fp16").to(device_1)
# PrefPaint
prefpaint_path = 'kd5678/prefpaint-v1.0'
prefpaint_pipe = AutoPipelineForInpainting.from_pretrained(
                        prefpaint_path,
                        torch_dtype=torch.float16,
                        safety_checker = None,
                        requires_safety_checker = False).to(device_1)
# IP Adapter
ip_checkpoint_path = "IP-Adapter/sdxl_models/ip-adapter_sdxl.bin"
image_encoder_path = "IP-Adapter/sdxl_models/image_encoder"
ip_model = IPAdapterXL(
                        diffusion_model, 
                        image_encoder_path, 
                        ip_checkpoint_path, 
                        device_1)
# Add-it
device_2 = "cuda:1" if torch.cuda.is_available() else "cpu"
my_transformer  = AdditFluxTransformer2DModel.from_pretrained("black-forest-labs/FLUX.1-dev", subfolder="transformer", torch_dtype=torch.float16)
addit_pipe = AdditFluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", 
                                      transformer=my_transformer,
                                      torch_dtype=torch.float16).to(device_2)
addit_pipe.scheduler = AdditFlowMatchEulerDiscreteScheduler.from_config(addit_pipe.scheduler.config)
# VTON
device_3 = "cuda:2" if torch.cuda.is_available() else "cpu"
vton_pipeline = CatVTONPipeline(
    base_ckpt="booksforcharlie/stable-diffusion-inpainting",
    attn_ckpt=snapshot_download(repo_id="zhengchong/CatVTON"),
    attn_ckpt_version="mix",
    weight_dtype=init_weight_dtype("fp16"),
    use_tf32=False,
    device=device_3
)
# SDXL
device_4 = "cuda:3" if torch.cuda.is_available() else "cpu"
sdxl_pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
sdxl_pipe.to(device_4)
# Object Removal
device_5 = "cuda:4" if torch.cuda.is_available() else "cpu"
generator = torch.Generator(device=device_5).manual_seed(42)
use_agf = True
removal_pipe = ObjectClearPipeline.from_pretrained_with_custom_modules(
    "jixin0101/ObjectClear",
    torch_dtype=torch.float16,
    apply_attention_guided_fusion=use_agf,
    cache_dir=None,
    variant="fp16",
).to(device_5)
# Llama4 Client
client_groq = Groq()
# GPT5 Client
client_gpt = OpenAI()