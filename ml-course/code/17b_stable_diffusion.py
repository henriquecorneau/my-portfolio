"""Lesson 17 (part 2) - Text-to-image with a pretrained latent diffusion model.

Needs a CUDA GPU with ~6 GB of memory (a free Colab T4 is enough) and:

    pip install diffusers transformers accelerate

This script is not run when the course figures are rebuilt (they are made on a CPU).
"""
import argparse

import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

# --8<-- [start:pipeline]
parser = argparse.ArgumentParser()
parser.add_argument("prompt", nargs="?", default="a robotic boat on a river at sunset, watercolor")
parser.add_argument("--model", default="stabilityai/stable-diffusion-2-1-base")   # any SD 1.x/2.x checkpoint
parser.add_argument("--steps", type=int, default=25)
parser.add_argument("--guidance", type=float, default=7.5)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", default="generated.png")
args = parser.parse_args()

pipe = StableDiffusionPipeline.from_pretrained(args.model, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)   # good images in ~25 steps
pipe = pipe.to("cuda")

generator = torch.Generator("cuda").manual_seed(args.seed)         # same seed + prompt = same image
image = pipe(args.prompt, num_inference_steps=args.steps, guidance_scale=args.guidance,
             generator=generator).images[0]                        # a PIL image, 512x512 for the base model
image.save(args.out)
print("saved", args.out)
# --8<-- [end:pipeline]
