import os
import sys
import re

with open("realtime_video.py", "r", encoding="utf-8") as f:
    code = f.read()

build_engine_str = '''
def load_model_and_engine(args, lora_name):
    from diffusers import AutoencoderTiny, StableDiffusionPipeline
    from streamdiffusion import StreamDiffusion
    from streamdiffusion.acceleration.tensorrt import accelerate_with_tensorrt
    
    log("[Engine] Loading base model (kohaku-v2.1)...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "KBlueLeaf/kohaku-v2.1",
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to("cuda")

    log("[Engine] Swapping in TinyVAE...")
    pipe.vae = AutoencoderTiny.from_pretrained(
        "madebyollin/taesd", torch_dtype=torch.float16
    ).to("cuda")

    engine_dir = os.path.join(SCRIPT_DIR, f"engines_tinyvae_fb{args.frame_buffer}")
    if lora_name and lora_name.lower() != "none":
        lora_path = os.path.join(SCRIPT_DIR, "loras", lora_name)
        if os.path.exists(lora_path):
            log(f"[Engine] Fusing character LoRA: {lora_name}")
            pipe.load_lora_weights(lora_path)
            pipe.fuse_lora()
            safe = "".join(c for c in lora_name if c.isalnum() or c in ("-", "_"))
            safe = safe.replace("safetensors", "")
            engine_dir = os.path.join(SCRIPT_DIR, f"engines_tinyvae_{safe}_fb{args.frame_buffer}")
        else:
            log(f"[Engine] LoRA not found at {lora_path} — continuing without it.")

    step1 = min(args.t_index - 2, max(10, args.t_index - 15))
    step1 = max(0, step1)
    t_list = [step1, args.t_index]
    
    stream = StreamDiffusion(
        pipe,
        t_index_list=t_list,
        torch_dtype=torch.float16,
        cfg_type="full",
        do_add_noise=True,
        use_denoising_batch=True,
        frame_buffer_size=args.frame_buffer,
    )

    log("[Engine] Loading LCM-LoRA...")
    stream.load_lcm_lora()
    stream.fuse_lora()
    
    log(f"[Engine] Applying TensorRT acceleration ({os.path.basename(engine_dir)}).")
    if not os.path.exists(os.path.join(engine_dir, "unet.engine")):
        log("[Engine] No cached engine found — the first build takes 5-15 minutes. Please wait.")
    stream = accelerate_with_tensorrt(
        stream,
        engine_dir,
        max_batch_size=stream.trt_unet_batch_size,
        use_cuda_graph=args.cuda_graph,
        engine_build_options={"opt_batch_size": stream.trt_unet_batch_size},
    )
    return pipe, stream

'''

# Inject load_model_and_engine above main()
code = code.replace('def main():', build_engine_str + '\\ndef main():')

target_start = '''    log("[Engine] Loading base model (kohaku-v2.1)...")
    pipe = StableDiffusionPipeline.from_pretrained('''

target_end = '''    stream = accelerate_with_tensorrt(
        stream,
        engine_dir,
        max_batch_size=stream.trt_unet_batch_size,
        use_cuda_graph=args.cuda_graph,
        engine_build_options={"opt_batch_size": stream.trt_unet_batch_size},
    )'''

match_start = code.find('    log("[Engine] Loading base model (kohaku-v2.1)...")')
match_end = code.find('engine_build_options={"opt_batch_size": stream.trt_unet_batch_size},\\n    )')

if match_start != -1 and match_end != -1:
    end_idx = match_end + len('engine_build_options={"opt_batch_size": stream.trt_unet_batch_size},\\n    )')
    
    new_init = '''    current_lora = args.lora
    pipe, stream = load_model_and_engine(args, current_lora)'''
    code = code[:match_start] + new_init + code[end_idx:]

loop_target = '''            if time.time() > warmup_until:
                stat_n += 1'''

hot_swap_code = '''
            if "lora" in state_dict and state_dict["lora"] != current_lora:
                new_lora = state_dict.pop("lora")
                log(f"[Engine] LoRA hot-swap requested: {current_lora} -> {new_lora}")
                current_lora = new_lora
                
                # Cleanup old engine
                import gc
                del stream
                del pipe
                gc.collect()
                torch.cuda.empty_cache()
                
                # Rebuild
                pipe, stream = load_model_and_engine(args, current_lora)
                state_dict["prompt_dirty"] = True
                
                # Re-warmup
                log("[Engine] Re-preparing TensorRT embeddings after hot-swap...")
                stream.prepare(
                    prompt=state_dict["base_prompt"] + (", closed mouth" if args.audio_sync else ""),
                    negative_prompt=state_dict["negative_prompt"],
                    num_inference_steps=50,
                    guidance_scale=args.guidance_scale,
                    delta=args.delta,
                )
                
            if time.time() > warmup_until:
                stat_n += 1'''

code = code.replace(loop_target, hot_swap_code.strip())

with open("realtime_video.py", "w", encoding="utf-8") as f:
    f.write(code)

print("done")