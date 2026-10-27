import os
import torch
from diffusers import DiffusionPipeline

MODEL_ID = os.environ.get("MODEL_ID", "OpenMOSS-Team/MOSS-Image-SFT")

if "pipe" not in globals() or pipe is None:
    print(f"Loading pipeline for {MODEL_ID}...")
    pipe = DiffusionPipeline.from_pretrained(
        MODEL_ID,
        custom_pipeline="pipeline_moss_image",
        trust_remote_code=True,
        torch_dtype={"default": torch.bfloat16, "vae": torch.float32},
    )
    pipe.scheduler = type(pipe.scheduler).from_config(pipe.scheduler.config)
    pipe.to("cuda")
    print("Pipeline loaded successfully onto CUDA!")

prompt = os.environ.get("PROMPT", "A lovely red panda eating bamboo in a misty forest, highly detailed, photorealistic")
steps = int(os.environ.get("STEPS", "25"))

print(f"Generating image with prompt: {prompt} (steps={steps})...")
image = pipe(
    prompt,
    height=1024,
    width=1024,
    num_inference_steps=steps,
    guidance_scale=3.0,
    uncond_skip_layers=[22],
    generator=torch.Generator("cpu").manual_seed(42),
).images[0]

output_path = "/content/test_output.png"
image.save(output_path)
print(f"Image saved successfully to {output_path}!")
