#%%
import diffusers
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from diffusers.utils import load_image, make_image_grid
from PIL import Image
import cv2
import numpy as np

import torch
import numpy as np

from transformers import pipeline
from diffusers.utils import load_image, make_image_grid

image = load_image(
    "/root/autodl-tmp/gaussian-splatting/data/truck/images/000001.jpg"
)

# def get_depth_map(image, depth_estimator):
#     image = depth_estimator(image)["depth"]
#     image = np.array(image)
#     image = image[:, :, None]
#     image = np.concatenate([image, image, image], axis=2)
#     detected_map = torch.from_numpy(image).float() / 255.0
#     depth_map = detected_map.permute(2, 0, 1)
#     return depth_map

# depth_estimator = pipeline("depth-estimation")
# depth_map = get_depth_map(image, depth_estimator).unsqueeze(0).half().to("cuda")

from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, UniPCMultistepScheduler
import torch

controlnet = ControlNetModel.from_pretrained("/root/autodl-tmp/local_models/contronet_depth", torch_dtype=torch.float16, use_safetensors=True)
pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16, use_safetensors=True
)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

output = pipe(
    "lego batman and robin", image=image, control_image=depth_map,
).images[0]
make_image_grid([image, output], rows=1, cols=2)

# import torch
# from diffusers import StableDiffusionPipeline
# from diffusers.utils import make_image_grid

# pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"
# repo_id_embeds = "sd-concepts-library/cat-toy"

# pipeline = StableDiffusionPipeline.from_pretrained(
#     pretrained_model_name_or_path, torch_dtype=torch.float16, use_safetensors=True
# ).to("cuda")

# pipeline.load_textual_inversion(repo_id_embeds)

# prompt = "a grafitti in a favela wall with a <cat-toy> on it"

# num_samples_per_row = 2
# num_rows = 2

# all_images = []
# for _ in range(num_rows):
#     images = pipeline(prompt, num_images_per_prompt=num_samples_per_row, num_inference_steps=50, guidance_scale=7.5).images
#     all_images.extend(images)

# grid = make_image_grid(all_images, num_rows, num_samples_per_row)
# grid.save("cat-toy.png")