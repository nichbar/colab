import os
import torch
import gradio as gr
from diffusers import DiffusionPipeline

MODEL_ID = "OpenMOSS-Team/MOSS-Image-SFT"

print("Checking pipeline in kernel memory...")
if "pipe" not in globals() or pipe is None:
    print(f"Loading {MODEL_ID} into GPU...")
    pipe = DiffusionPipeline.from_pretrained(
        MODEL_ID,
        custom_pipeline="pipeline_moss_image",
        trust_remote_code=True,
        torch_dtype={"default": torch.bfloat16, "vae": torch.float32},
    )
    pipe.scheduler = type(pipe.scheduler).from_config(pipe.scheduler.config)
    pipe.to("cuda")
    print("Pipeline loaded!")
else:
    print("Reusing existing pipeline already in memory!")

def generate(prompt, negative_prompt, width, height, steps, guidance_scale, seed, skip_layer_22):
    if not prompt or not prompt.strip():
        raise gr.Error("Please enter a prompt.")
    
    gen = None
    if seed is not None and int(seed) >= 0:
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        
    w = max(256, (int(width) // 16) * 16)
    h = max(256, (int(height) // 16) * 16)
    uncond_skip = [22] if skip_layer_22 else []
    # Note: MOSS-Image custom pipeline expects negative_prompt to be a string (e.g. ""), NOT None.
    neg = str(negative_prompt).strip() if negative_prompt is not None else ""

    print(f"[Generate] Prompt: '{prompt[:60]}...' | Neg: '{neg}' | Resolution: {w}x{h} | Steps: {steps} | CFG: {guidance_scale} | Seed: {seed}")
    try:
        output = pipe(
            prompt=prompt.strip(),
            negative_prompt=neg,
            height=h,
            width=w,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance_scale),
            uncond_skip_layers=uncond_skip,
            generator=gen,
        )
        return output.images[0]
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise gr.Error(f"Generation error: {str(e)}")

# Close any previous instances
gr.close_all()

with gr.Blocks(title="MOSS-Image-SFT Demo") as demo:
    gr.Markdown("# 🌸 MOSS-Image-SFT (Colab Pro A100)")
    gr.Markdown(
        "Supervised Fine-Tuned (SFT) text-to-image diffusion model from OpenMOSS-Team. "
        "Supports bilingual English/Chinese prompts, detailed text rendering, and high-fidelity generation."
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(
                label="Prompt",
                lines=4,
                placeholder="Describe the image you want to generate (English or Chinese supported)...",
                value='A wide view of a refrigerator beside a kitchen counter with a fruit bowl. A small sticky note on the refrigerator door is handwritten in Chinese:\n"记得买牛奶和鸡蛋！\n牛奶两盒，鸡蛋一打。\n顺便带一袋面包。"',
            )
            negative_prompt = gr.Textbox(
                label="Negative Prompt",
                lines=2,
                placeholder="Blurry, low quality, bad anatomy (leave blank if none)...",
                value="",
            )
            
            with gr.Row():
                width = gr.Slider(minimum=512, maximum=1536, step=16, value=1024, label="Width")
                height = gr.Slider(minimum=512, maximum=1536, step=16, value=1024, label="Height")
                
            with gr.Row():
                steps = gr.Slider(minimum=10, maximum=100, step=1, value=50, label="Steps (50 recommended)")
                guidance_scale = gr.Slider(minimum=1.0, maximum=10.0, step=0.5, value=3.0, label="CFG Guidance Scale (3.0 recommended)")
                
            with gr.Row():
                seed = gr.Number(value=42, label="Seed (-1 for random)", precision=0)
                skip_layer_22 = gr.Checkbox(value=True, label="Skip Layer 22 (Recommended 2x speedup for negative branch)")
                
            btn = gr.Button("Generate Image", variant="primary")
            
        with gr.Column(scale=1):
            out_img = gr.Image(label="Generated Image", type="pil")

    gr.Examples(
        examples=[
            [
                'A wide view of a refrigerator beside a kitchen counter with a fruit bowl. A small sticky note on the refrigerator door is handwritten in Chinese:\n"记得买牛奶和鸡蛋！\n牛奶两盒，鸡蛋一打。\n顺便带一袋面包。"',
                "",
                1024,
                1024,
                50,
                3.0,
                42,
                True,
            ],
            [
                "A cute fluffy red panda wearing green explorer clothing and a small backpack, standing on a mossy fallen log in an ancient Redwood forest, soft cinematic sunlight, masterpiece, 8k resolution.",
                "",
                1024,
                1024,
                50,
                3.0,
                1234,
                True,
            ],
            [
                "一只戴着金丝眼镜的英国短毛猫坐在精致的书房书桌前，桌上摆着古老的羊皮纸书和热气腾腾的咖啡，写实摄影风格，电影光影。",
                "",
                1024,
                1024,
                50,
                3.0,
                2026,
                True,
            ],
        ],
        inputs=[prompt, negative_prompt, width, height, steps, guidance_scale, seed, skip_layer_22],
    )
            
    btn.click(
        fn=generate,
        inputs=[prompt, negative_prompt, width, height, steps, guidance_scale, seed, skip_layer_22],
        outputs=out_img,
    )

_, local_url, share_url = demo.queue().launch(share=True, prevent_thread_lock=True)
print(f"GRADIO_PUBLIC_URL={share_url}")
print(f"GRADIO_LOCAL_URL={local_url}")
