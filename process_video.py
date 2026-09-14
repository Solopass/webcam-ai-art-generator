import os
import sys
import time
import json
import numpy as np
import cv2
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from realtime_video import (
    build_args, load_model_and_engine, refit_lora_to_trt
)

def process_video():
    settings = {}
    if os.path.exists("vtuber_settings.json"):
        try:
            with open("vtuber_settings.json", "r") as f:
                settings = json.load(f)
        except Exception:
            pass
            
    # argv[1] wins: the launcher's Export button validates a path and now
    # passes it. Before this the button read a path, checked it, threw it away
    # and rendered whatever "camera" happened to be in the settings file.
    video_path = sys.argv[1] if len(sys.argv) > 1 else settings.get("camera", "")
    if not os.path.isfile(video_path):
        print(f"[Error] The camera setting '{video_path}' is not a valid video file.")
        return
        
    output_path = "output_vfx.mp4"
    
    args = build_args(raw_args=[])
    args.prompt = settings.get("prompt", args.prompt)
    args.negative_prompt = settings.get("negative_prompt", args.negative_prompt)
    args.lora = settings.get("lora", args.lora)
    args.guidance_scale = settings.get("guidance", args.guidance_scale)
    args.t_index = int(settings.get("ai_strength", args.t_index))
    args.post_processing = settings.get("post_processing", args.post_processing)
    args.composite = settings.get("composite", args.composite)
    args.no_face_track = settings.get("no_face_track", False)
    
    vfx_blend_mode = settings.get("vfx_blend_mode", "Normal")
    vfx_opacity = settings.get("vfx_opacity", 1.0)
    zoom = settings.get("zoom", 1.0)
    
    print(f"[Export] Loading video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Error] Failed to open {video_path}")
        return
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"[Export] Loading TensorRT Engine and Models...")
    pipe, stream, controlnet_aux_models = load_model_and_engine(args, args.lora)
    if args.lora and args.lora.lower() != "none":
        refit_lora_to_trt(stream, args, args.lora)
        
    stream.prepare(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=50,
        guidance_scale=args.guidance_scale,
        delta=args.delta,
    )
    
    segmenter = None
    if args.composite:
        import mediapipe.python.solutions as mp_solutions
        segmenter = mp_solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
        
    print(f"[Export] Starting Offline Render: {total_frames} frames...")
    
    for _ in range(3):
        x_in = torch.randn((1, 3, 512, 512), dtype=torch.float16, device="cuda")
        stream(x_in)
        
    import mediapipe as mp
    mp_face_detection = mp.solutions.face_detection
    face_detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    
    frame_idx = 0
    start_time = time.time()
    
    current_x, current_y, current_size = None, None, None
    smooth_factor = 0.2
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        full_frame = frame.copy()
        h, w = frame.shape[:2]
        size = min(h, w)
        box_x = (w - size) // 2
        box_y = (h - size) // 2
        
        results = face_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.detections:
            det = results.detections[0]
            bboxC = det.location_data.relative_bounding_box
            cx = int((bboxC.xmin + bboxC.width / 2) * w)
            cy = int((bboxC.ymin + bboxC.height / 2) * h)
            
            size = min(h, w) / zoom
            box_x = int(cx - size / 2)
            box_y = int(cy - size / 2)
            
        box_x = max(0, min(box_x, w - size))
        box_y = max(0, min(box_y, h - size))
        
        if current_x is None:
            current_x, current_y, current_size = box_x, box_y, size
        else:
            current_x = current_x + (box_x - current_x) * smooth_factor
            current_y = current_y + (box_y - current_y) * smooth_factor
            current_size = current_size + (size - current_size) * smooth_factor
            
        x, y, s = int(current_x), int(current_y), int(current_size)
        
        if args.no_face_track:
            x, y = 0, 0
            cw, ch = w, h
            cropped = frame
        else:
            cw, ch = s, s
            cropped = frame[y:y+ch, x:x+cw]
            
        if cropped.size == 0:
            out.write(frame)
            continue
            
        resized = cv2.resize(cropped, (512, 512))
        frame_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        original_frame_rgb = frame_rgb.copy()
        
        soft_mask = None
        if segmenter is not None:
            small_rgb = cv2.resize(original_frame_rgb, (256, 256), interpolation=cv2.INTER_LINEAR)
            seg_res = segmenter.process(small_rgb)
            if seg_res.segmentation_mask is not None:
                mask = cv2.resize(seg_res.segmentation_mask, (512, 512), interpolation=cv2.INTER_LINEAR)
                soft_mask = mask.copy()
                mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)
                mask = (mask > 0.3).astype(np.float32)
                mask = cv2.GaussianBlur(mask, (7, 7), 0)
                mask_3d = mask[..., np.newaxis]
                frame_rgb = (frame_rgb * mask_3d).astype(np.uint8)
                
        cond_final = None
        if len(controlnet_aux_models) > 0:
            cond_tensors = []
            for net_type in ["depth", "canny", "lineart", "openpose"]:
                if net_type in controlnet_aux_models:
                    aux = controlnet_aux_models[net_type]
                    with torch.no_grad():
                        cond_pil = aux(Image.fromarray(frame_rgb))
                    c_tensor = TF.to_tensor(cond_pil).unsqueeze(0).to(dtype=torch.float16, device="cuda")
                    if soft_mask is not None:
                        mask_tensor = torch.from_numpy(soft_mask).unsqueeze(0).unsqueeze(0).to(device=c_tensor.device, dtype=c_tensor.dtype)
                        c_tensor = c_tensor * mask_tensor
                    cond_tensors.append(c_tensor)
            if len(cond_tensors) > 0:
                cond_final = torch.cat(cond_tensors, dim=1)
                
        x_in = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0).to("cuda", dtype=torch.float16) / 255.0
        
        kwargs = {}
        if cond_final is not None:
            kwargs["controlnet_cond"] = cond_final
            
        output_image = stream(x_in, **kwargs)
        
        if output_image is None:
            continue
            
        if isinstance(output_image, np.ndarray):
            out_np = output_image
            if out_np.dtype != np.uint8:
                out_np = (np.clip(out_np, 0.0, 1.0) * 255).astype(np.uint8)
            out_frame = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)
        else:
            out_t = output_image[-1] if len(output_image.shape) == 4 else output_image
            out_t = (out_t / 2 + 0.5).clamp(0, 1)
            out_t = (out_t * 255).byte().permute(1, 2, 0)
            out_np = out_t.cpu().numpy()
            out_frame = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)
            
        if args.post_processing:
            gaussian = cv2.GaussianBlur(out_frame, (0, 0), 1.5)
            out_frame = cv2.addWeighted(out_frame, 1.4, gaussian, -0.4, 0)
            hsv = cv2.cvtColor(out_frame, cv2.COLOR_BGR2HSV)
            hh, ss, vv = cv2.split(hsv)
            ss = cv2.add(ss, 20)
            vv = cv2.add(vv, 10)
            out_frame = cv2.cvtColor(cv2.merge((hh, ss, vv)), cv2.COLOR_HSV2BGR)
            
        cw = min(cw, w - x)
        ch = min(ch, h - y)
        if cw > 0 and ch > 0:
            ai_resized = cv2.resize(out_frame, (cw, ch))
            hd_region = full_frame[y:y+ch, x:x+cw].copy()
            
            mask_resized = None
            if soft_mask is not None:
                mask_resized = cv2.resize(soft_mask, (cw, ch))[..., np.newaxis]
                
            base_region = hd_region.astype(np.float32)
            ai_region = ai_resized.astype(np.float32)
            
            if vfx_blend_mode == "Screen":
                blended = 255.0 - ((255.0 - base_region) * (255.0 - ai_region) / 255.0)
            elif vfx_blend_mode == "Linear Dodge (Add)":
                blended = np.clip(base_region + ai_region, 0, 255)
            elif vfx_blend_mode == "Color Dodge":
                blended = np.clip(base_region / (1.0001 - ai_region/255.0), 0, 255)
            elif vfx_blend_mode == "Overlay":
                mask = base_region < 128
                blended = np.empty_like(base_region)
                blended[mask] = 2.0 * base_region[mask] * ai_region[mask] / 255.0
                blended[~mask] = 255.0 - 2.0 * (255.0 - base_region[~mask]) * (255.0 - ai_region[~mask]) / 255.0
            elif vfx_blend_mode == "Hard Light":
                mask = ai_region < 128
                blended = np.empty_like(base_region)
                blended[mask] = 2.0 * base_region[mask] * ai_region[mask] / 255.0
                blended[~mask] = 255.0 - 2.0 * (255.0 - base_region[~mask]) * (255.0 - ai_region[~mask]) / 255.0
            elif vfx_blend_mode == "Soft Light":
                B_norm = base_region / 255.0
                A_norm = ai_region / 255.0
                mask = A_norm <= 0.5
                blended = np.empty_like(base_region)
                blended[mask] = B_norm[mask] - (1.0 - 2.0 * A_norm[mask]) * B_norm[mask] * (1.0 - B_norm[mask])
                blended[~mask] = B_norm[~mask] + (2.0 * A_norm[~mask] - 1.0) * (np.sqrt(B_norm[~mask]) - B_norm[~mask])
                blended *= 255.0
            elif vfx_blend_mode == "Multiply":
                blended = (base_region * ai_region) / 255.0
            elif vfx_blend_mode == "Darken":
                blended = np.minimum(base_region, ai_region)
            elif vfx_blend_mode == "Lighten":
                blended = np.maximum(base_region, ai_region)
            elif vfx_blend_mode == "Difference":
                blended = np.abs(base_region - ai_region)
            elif vfx_blend_mode == "Exclusion":
                blended = base_region + ai_region - (2.0 * base_region * ai_region) / 255.0
            else:
                blended = ai_region
                
            final_ai = (blended * vfx_opacity) + (base_region * (1.0 - vfx_opacity))
            
            if mask_resized is not None:
                final_ai = (final_ai * mask_resized) + (base_region * (1.0 - mask_resized))
                
            full_frame[y:y+ch, x:x+cw] = final_ai.astype(np.uint8)
            
        out.write(full_frame)
        
        frame_idx += 1
        elapsed = time.time() - start_time
        fps_current = frame_idx / elapsed
        print(f"\r[Export] Rendered {frame_idx}/{total_frames} frames ({fps_current:.1f} fps) | Output: {output_path}", end="")
        
    while True:
        x_in = torch.randn((1, 3, 512, 512), dtype=torch.float16, device="cuda")
        output_image = stream(x_in)
        if output_image is None or frame_idx >= total_frames:
            break
        out.write(full_frame)
        frame_idx += 1

    print(f"\n[Export] DONE! Saved to {output_path}")
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_video()
