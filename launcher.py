import json
import os
import subprocess
import sys
import threading
import time
import tkinter as tk

import customtkinter as ctk

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(SCRIPT_DIR, "vtuber_settings.json")
LOG_DIR = os.path.join(SCRIPT_DIR, "logs")

DEFAULTS = {
    "prompt": "1girl, masterpiece, ultra-detailed, cinematic lighting, vibrant colors, "
              "flat color, anime key visual, simple background",
    "negative_prompt": "blurry, deformed, bad anatomy, pixelated, jpeg artifacts",
    "camera": "0",
    "embedded_preview": True,
    "mirror_camera": True,
    "virtual_camera": False,
    "audio_sync": False,
    "keep_background": False,
    "normalize_lighting": False,
    "cuda_graph": False,
    "lora": "None",
    "bg_image": "",
    "guidance": 1.4,
    "delta": 1.0,
    "ai_strength": 40.0,
    "freeze": 1.0,
}

# AI Strength 0-100 maps onto the StreamDiffusion denoise start step.
# Low t_index = the model starts from heavy noise = heavy stylisation.
T_INDEX_MIN, T_INDEX_MAX = 12, 45


def strength_to_t_index(strength):
    strength = max(0.0, min(100.0, float(strength)))
    return int(round(T_INDEX_MAX - (strength / 100.0) * (T_INDEX_MAX - T_INDEX_MIN)))


class VTuberStudioApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Antigravity VTuber Studio")
        self.geometry("1200x900")
        self.minsize(900, 700)
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.process = None
        self.current_frame_image = None
        self.log_file = None
        import collections
        self.frame_buffer = collections.deque(maxlen=150)

        # Global Keybinds
        self.bind("<Control-s>", lambda e: self.save_replay())
        self.bind("<Control-r>", lambda e: self.randomize_prompt())
        self.bind("<F12>", lambda e: self.take_snapshot())

        # Threading for non-blocking save
        self.saving = False
        
        # Continuous Recording
        self.is_recording = False
        self.video_writer = None
        try:
            os.makedirs(LOG_DIR, exist_ok=True)
            self.log_file = open(os.path.join(LOG_DIR, "launcher-latest.log"),
                                 "w", encoding="utf-8", buffering=1)
        except Exception:
            pass
        # NOTE: this used to be self.config, which shadows Tk's own
        # Misc.config() method and breaks CustomTkinter's internal calls.
        self.settings = self.load_settings()

        self.build_ui()
        self.apply_settings()

    # ------------------------------------------------------------ config --
    def load_settings(self):
        settings = DEFAULTS.copy()
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                if isinstance(saved, dict):
                    settings.update(saved)
            except Exception:
                pass
        return settings

    def save_settings(self):
        settings = {
            "prompt": self.prompt_entry.get(),
            "negative_prompt": self.neg_prompt_entry.get(),
            "camera": self.camera_entry.get(),
            "embedded_preview": self.preview_var.get(),
            "mirror_camera": self.mirror_var.get(),
            "virtual_camera": self.vcam_var.get(),
            "audio_sync": self.audio_var.get(),
            "keep_background": self.bg_keep_var.get(),
            "normalize_lighting": self.clahe_var.get(),
            "cuda_graph": self.cudagraph_var.get(),
            "lora": self.lora_var.get(),
            "bg_image": self.bg_var.get(),
            "guidance": self.guidance_var.get(),
            "ai_strength": self.strength_var.get(),
            "freeze": self.freeze_var.get(),
            "motion_smoothing": self.motion_var.get(),
            "bokeh_blur": self.bokeh_var.get(),
            "normalize_lighting": self.clahe_var.get(),
            "cuda_graph": self.cudagraph_var.get()
        }
        for e, v in self.expr_vars.items():
            settings[f"expr_{e}"] = v.get()
        for k, v in self.sens_vars.items():
            settings[f"sens_{k}"] = v.get()
            
        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(settings, f, indent=4)
        except Exception as e:
            self.log(f"Could not save settings: {e}")

    # ---------------------------------------------------------------- UI --
    def build_ui(self):
        self.header = ctk.CTkLabel(self, text="AI VTuber Studio",
                                   font=ctk.CTkFont(size=24, weight="bold"))
        self.header.pack(pady=(20, 10))

        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.pack(fill=ctk.BOTH, expand=True, padx=20, pady=10)

        self.left_col = ctk.CTkFrame(self.main_frame)
        self.left_col.pack(side=ctk.LEFT, fill=ctk.BOTH, expand=True, padx=(0, 10))

        self.right_col = ctk.CTkScrollableFrame(self.main_frame, width=290)
        self.right_col.pack(side=ctk.RIGHT, fill=ctk.Y)

        def build_prompt_header(label_text, entry_box, settings_key):
            header = ctk.CTkFrame(self.left_col, fg_color="transparent")
            header.pack(fill=ctk.X, padx=10, pady=(10, 0))
            saved = self.settings.get(settings_key, [])
            
            def on_select(val):
                if val != label_text:
                    entry_box.delete(0, 'end')
                    entry_box.insert(0, val)
                    self.apply_prompt()
                dropdown.set(label_text)
                
            def on_save():
                current = entry_box.get().strip()
                if current and current not in saved:
                    saved.append(current)
                    self.settings[settings_key] = saved
                    self.save_settings()
                    dropdown.configure(values=[label_text] + saved)
                    
            dropdown = ctk.CTkOptionMenu(header, values=[label_text] + saved, command=on_select,
                                         text_color=("black", "white"))
            dropdown.set(label_text)
            dropdown.pack(side=ctk.LEFT)
            ctk.CTkButton(header, text="💾", width=30, height=24, fg_color="transparent",
                          command=on_save, hover_color="#333333").pack(side=ctk.LEFT, padx=5)

        # --- LEFT: prompting + preview ---
        prompt_row = ctk.CTkFrame(self.left_col, fg_color="transparent")
        
        self.prompt_entry = ctk.CTkEntry(prompt_row, placeholder_text="Describe your VTuber...")
        build_prompt_header("Master Prompt ▾", self.prompt_entry, "saved_prompts")
        prompt_row.pack(fill=ctk.X, padx=10, pady=5)
        self.prompt_entry.pack(side=ctk.LEFT, fill=ctk.X, expand=True)
        self.apply_btn = ctk.CTkButton(prompt_row, text="Apply ⏎", width=86,
                                       command=self.apply_prompt)
        self.apply_btn.pack(side=ctk.LEFT, padx=(6, 0))

        self.neg_prompt_entry = ctk.CTkEntry(self.left_col)
        build_prompt_header("Negative Prompt ▾", self.neg_prompt_entry, "saved_neg_prompts")
        self.neg_prompt_entry.pack(fill=ctk.X, padx=10, pady=5)

        # Pressing Enter in either box pushes the text to a running engine over
        # the same command channel the randomizer uses. Without this the boxes
        # were read only once, when the engine was launched.
        self.prompt_entry.bind("<Return>", lambda e: self.apply_prompt())
        self.neg_prompt_entry.bind("<Return>", lambda e: self.apply_prompt())

        self.video_frame = ctk.CTkFrame(self.left_col, fg_color="black")
        self.video_frame.pack(fill=ctk.BOTH, expand=True, padx=10, pady=10)
        self.video_label = tk.Label(self.video_frame, text="Engine stopped.",
                                    bg="black", fg="white", font=("Arial", 12))
        self.video_label.pack(expand=True, fill=tk.BOTH)

        # --- RIGHT: settings ---
        ctk.CTkLabel(self.right_col, text="Studio Settings",
                     font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)

        self.settings_frame = ctk.CTkFrame(self.right_col, fg_color="transparent")
        self.settings_frame.pack(fill=ctk.X, padx=10)

        ctk.CTkLabel(self.settings_frame, text="Camera Index:", anchor="w").grid(
            row=0, column=0, padx=5, pady=5, sticky="w")
        self.camera_entry = ctk.CTkComboBox(self.settings_frame, width=150, values=["0", "1", "screen 1 (Primary)", "screen 2 (Secondary)", "screen 3 (Tertiary)"])
        self.camera_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        ctk.CTkLabel(self.settings_frame, text="Character (LoRA):", anchor="w").grid(
            row=1, column=0, padx=5, pady=5, sticky="w")
        lora_dir = os.path.join(SCRIPT_DIR, "loras")
        loras = ["None (Original Default)"]
        if os.path.isdir(lora_dir):
            loras += sorted(f for f in os.listdir(lora_dir) if f.endswith(".safetensors"))
        def on_lora_changed(val):
            if val == "None (Original Default)":
                val = "None"
            if self.process is not None:
                self.send_command({"lora": val})

        saved_lora = self.settings.get("lora", "None (Original Default)")
        if saved_lora == "None":
            saved_lora = "None (Original Default)"
        self.lora_var = ctk.StringVar(value=saved_lora if saved_lora in loras else "None (Original Default)")
        self.lora_dropdown = ctk.CTkOptionMenu(self.settings_frame, variable=self.lora_var,
                                               values=loras, width=140, command=on_lora_changed)
        self.lora_dropdown.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        # Toggles
        self.preview_var = ctk.BooleanVar(value=self.settings.get("embedded_preview", True))
        self.preview_cb = ctk.CTkSwitch(self.right_col, text="Embedded Video Preview",
                                        command=self.toggle_preview, variable=self.preview_var)
        self.preview_cb.pack(anchor="w", padx=15, pady=4)

        self.mirror_var = ctk.BooleanVar(value=self.settings.get("mirror_camera", True))
        self.mirror_cb = ctk.CTkSwitch(self.right_col, text="Mirror Camera",
                                       variable=self.mirror_var)
        self.mirror_cb.pack(anchor="w", padx=15, pady=4)

        self.vcam_var = ctk.BooleanVar(value=self.settings.get("virtual_camera", False))
        self.vcam_cb = ctk.CTkSwitch(self.right_col, text="OBS Virtual Camera",
                                     variable=self.vcam_var)
        self.vcam_cb.pack(anchor="w", padx=15, pady=4)

        self.audio_var = ctk.BooleanVar(value=self.settings.get("audio_sync", False))
        self.audio_cb = ctk.CTkSwitch(self.right_col, text="Audio Lip-Sync (FFT)",
                                      variable=self.audio_var)
        self.audio_cb.pack(anchor="w", padx=15, pady=4)

        self.bg_keep_var = ctk.BooleanVar(value=self.settings.get("keep_background", False))
        self.bg_keep_cb = ctk.CTkSwitch(self.right_col, text="Keep Real Background",
                                        variable=self.bg_keep_var)
        self.bg_keep_cb.pack(anchor="w", padx=15, pady=4)

        # Background image
        self.bg_frame = ctk.CTkFrame(self.right_col, fg_color="transparent")
        self.bg_frame.pack(fill=ctk.X, padx=10, pady=(5, 0))
        ctk.CTkLabel(self.bg_frame, text="BG:", anchor="w").pack(side=ctk.LEFT, padx=5)
        self.bg_var = ctk.StringVar(value=self.settings.get("bg_image", ""))
        self.bg_entry = ctk.CTkEntry(self.bg_frame, textvariable=self.bg_var, width=150)
        self.bg_entry.pack(side=ctk.LEFT, padx=5)
        self.bg_btn = ctk.CTkButton(self.bg_frame, text="Browse", width=60, command=self.select_bg)
        self.bg_btn.pack(side=ctk.LEFT, padx=(0, 5))
        self.bg_clear_btn = ctk.CTkButton(self.bg_frame, text="X", width=20, command=lambda: self.bg_var.set(""))
        self.bg_clear_btn.pack(side=ctk.LEFT)

        # --- Advanced tuning ---
        ctk.CTkLabel(self.right_col, text="Advanced Tuning:", anchor="w",
                     font=ctk.CTkFont(size=14, weight="bold")).pack(
            fill=ctk.X, padx=10, pady=(15, 5))

        self.tune_frame = ctk.CTkFrame(self.right_col, fg_color="transparent")
        self.tune_frame.pack(fill=ctk.X, padx=10)
        self.tune_frame.grid_columnconfigure(1, weight=1)

        self.strength_var = ctk.DoubleVar(value=self.settings.get("ai_strength", 40.0))
        self._slider_row(0, "AI Strength", self.strength_var, 0, 100, 20,
                         fmt=lambda v: f"{int(v)} (step {strength_to_t_index(v)})",
                         desc="Higher = closer to raw webcam, Lower = more AI stylization.")

        # Floor of 1.05, not 1.0: at exactly 1.0 StreamDiffusion disables
        # classifier-free guidance entirely, which halves the prompt-embed
        # tensor and changes the UNet batch out from under the built engine.
        self.guidance_var = ctk.DoubleVar(value=max(1.05, self.settings.get("guidance", 1.4)))
        self._slider_row(1, "Prompt Strictness (CFG)", self.guidance_var, 1.05, 3.0, 20, fmt=lambda v: f"{v:.2f}",
                         desc="How strictly the AI follows your text prompt. High values may look deep-fried.")

        import json
        
        def _send_freeze(v):
            if getattr(self, "cmd_socket", None):
                try: self.cmd_socket.send_string(json.dumps({"freeze_threshold": float(v)}))
                except Exception: pass
                
        def _send_motion(v):
            if getattr(self, "cmd_socket", None):
                try: self.cmd_socket.send_string(json.dumps({"motion_smoothing": float(v)}))
                except Exception: pass

        self.freeze_var = ctk.DoubleVar(value=self.settings.get("freeze", 1.0))
        self._slider_row(2, "Freeze Filter", self.freeze_var, 0.90, 1.00, 10,
                         fmt=lambda v: "off" if v >= 0.999 else f"{v:.2f}",
                         on_change=_send_freeze,
                         desc="Pauses generation when you sit perfectly still to increase visual quality.")

        self.motion_var = ctk.DoubleVar(value=self.settings.get("motion_smoothing", 0.6))
        self._slider_row(3, "Motion Blur", self.motion_var, 0.0, 0.9, 90,
                         fmt=lambda v: "off" if v < 0.01 else f"{v:.2f}",
                         on_change=_send_motion,
                         desc="Blends frames together for cinematic motion blur. Set to 'off' for raw responsiveness.")

        def _send_bokeh(v):
            if getattr(self, "cmd_socket", None):
                try: self.cmd_socket.send_string(json.dumps({"bokeh_blur": float(v)}))
                except Exception: pass

        self.bokeh_var = ctk.DoubleVar(value=self.settings.get("bokeh_blur", 0.0))
        self._slider_row(4, "Background Bokeh", self.bokeh_var, 0.0, 1.0, 100,
                         fmt=lambda v: "off" if v < 0.01 else f"{v:.2f}",
                         on_change=_send_bokeh,
                         desc="Artificially blurs the real room behind the AI character (only works if 'Composite Real Background' is checked).")

        self.clahe_var = ctk.BooleanVar(value=self.settings.get("normalize_lighting", False))
        self.clahe_cb = ctk.CTkSwitch(self.right_col, text="Normalize Lighting (CLAHE)",
                                      variable=self.clahe_var)
        self.clahe_cb.pack(anchor="w", padx=15, pady=(8, 4))

        self.cudagraph_var = ctk.BooleanVar(value=self.settings.get("cuda_graph", True))
        self.cudagraph_cb = ctk.CTkSwitch(self.right_col, text="Use CUDA Graphs",
                                          variable=self.cudagraph_var)
        self.cudagraph_cb.pack(anchor="w", padx=15, pady=4)

        # Expressions
        ctk.CTkLabel(self.right_col, text="Expression Overrides",
                     font=ctk.CTkFont(size=14, weight="bold")).pack(fill=ctk.X, padx=10, pady=(15, 5))
        
        self.expr_frame = ctk.CTkFrame(self.right_col, fg_color="transparent")
        self.expr_frame.pack(fill=ctk.X, padx=10)
        
        self.expr_vars = {}
        for i, expr in enumerate(["smiling", "open mouth", "closed eyes"]):
            ctk.CTkLabel(self.expr_frame, text=f"{expr}:", font=ctk.CTkFont(size=11), anchor="w").grid(row=i*2, column=0, sticky="w", pady=(5,0))
            var = ctk.StringVar(value=self.settings.get(f"expr_{expr}", expr))
            self.expr_vars[expr] = var
            entry = ctk.CTkEntry(self.expr_frame, textvariable=var, width=250, height=24)
            entry.grid(row=i*2+1, column=0, sticky="ew")
            # Bind live update over ZMQ
            def make_cb(e, v):
                return lambda *args: self.cmd_socket.send_string(json.dumps({"expr_override": {e: v.get()}})) if getattr(self, "cmd_socket", None) else None
            var.trace_add("write", make_cb(expr, var))

        # Sensitivities
        ctk.CTkLabel(self.right_col, text="Trigger Sensitivities",
                     font=ctk.CTkFont(size=14, weight="bold")).pack(fill=ctk.X, padx=10, pady=(15, 5))
        
        self.sens_frame = ctk.CTkFrame(self.right_col, fg_color="transparent")
        self.sens_frame.pack(fill=ctk.X, padx=10)
        
        self.sens_vars = {}
        defaults = {"smile": 0.010, "mouth": 0.030, "eyes": 0.019, "audio": 1.5}
        limits = {"smile": (0.001, 0.05), "mouth": (0.005, 0.1), "eyes": (0.005, 0.05), "audio": (0.1, 10.0)}
        
        for i, key in enumerate(["smile", "mouth", "eyes", "audio"]):
            ctk.CTkLabel(self.sens_frame, text=f"{key}:", font=ctk.CTkFont(size=11), anchor="w").grid(row=i, column=0, sticky="w", pady=2)
            var = ctk.DoubleVar(value=self.settings.get(f"sens_{key}", defaults[key]))
            self.sens_vars[key] = var
            
            def make_sens_cb(k, v):
                return lambda val: self.cmd_socket.send_string(json.dumps({"sens_override": {k: float(val)}})) if getattr(self, "cmd_socket", None) else None
                
            slider = ctk.CTkSlider(self.sens_frame, variable=var, from_=limits[key][0], to=limits[key][1], width=140, command=make_sens_cb(key, var))
            slider.grid(row=i, column=1, sticky="ew", padx=(10, 0), pady=2)

        # Logs
        ctk.CTkLabel(self.right_col, text="Engine Logs:", anchor="w").pack(
            fill=ctk.X, padx=10, pady=(10, 0))
        self.log_box = ctk.CTkTextbox(self.right_col, state="disabled", wrap="word",
                                      fg_color="#1E1E1E", height=140)
        self.log_box.pack(fill=ctk.BOTH, expand=True, padx=10, pady=(2, 5))

        # ZMQ preview subscriber. CONFLATE keeps only the newest frame, so a
        # slow GUI can never build up a backlog of stale frames.
        import zmq
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.SUB)
        self.zmq_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.zmq_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_socket.setsockopt(zmq.LINGER, 0)

        self.random_btn = ctk.CTkButton(self.right_col, text="🎲 Randomize Style (Ctrl+R)",
                                        command=self.randomize_prompt)
        self.random_btn.pack(fill=ctk.X, padx=10, pady=5)
        
        self.snapshot_btn = ctk.CTkButton(self.right_col, text="🖼️ Take Snapshot (F12)",
                                          command=self.take_snapshot)
        self.snapshot_btn.pack(fill=ctk.X, padx=10, pady=5)
        
        self.replay_btn = ctk.CTkButton(self.right_col, text="📷 Save 5s Replay (Ctrl+S)",
                                        command=self.save_replay)
        self.replay_btn.pack(fill=ctk.X, padx=10, pady=5)
        
        self.record_btn = ctk.CTkButton(self.right_col, text="🔴 Start Recording",
                                        command=self.toggle_recording, fg_color="#d9534f", hover_color="#c9302c")
        self.record_btn.pack(fill=ctk.X, padx=10, pady=5)

        self.start_btn = ctk.CTkButton(self.right_col, text="▶ START ENGINE",
                                       fg_color="#28a745", hover_color="#218838",
                                       command=self.start_script)
        self.start_btn.pack(fill=ctk.X, padx=10, pady=(20, 8))

        self.stop_btn = ctk.CTkButton(self.right_col, text="■ STOP", fg_color="#dc3545",
                                      hover_color="#c82333", state="disabled",
                                      command=self.stop_script)
        self.stop_btn.pack(fill=ctk.X, padx=10, pady=(0, 10))

        self.toggle_preview()
        self.update_video_frame()

    def _slider_row(self, row, label, var, lo, hi, steps, fmt, desc=None, on_change=None):
        r = row * 2
        ctk.CTkLabel(self.tune_frame, text=f"{label}:", anchor="w").grid(
            row=r, column=0, padx=2, pady=3, sticky="w")
        value_label = ctk.CTkLabel(self.tune_frame, text=fmt(var.get()), width=90, anchor="e")
        value_label.grid(row=r, column=2, padx=2, pady=3, sticky="e")
        
        def _update(v):
            value_label.configure(text=fmt(v))
            if on_change:
                on_change(v)
                
        slider = ctk.CTkSlider(self.tune_frame, variable=var, from_=lo, to=hi,
                               number_of_steps=steps, width=110, command=_update)
        slider.grid(row=r, column=1, padx=2, pady=3, sticky="ew")
        
        if desc:
            desc_label = ctk.CTkLabel(self.tune_frame, text=desc, font=ctk.CTkFont(size=11), text_color="gray", justify="left", wraplength=350)
            desc_label.grid(row=r+1, column=0, columnspan=3, padx=2, pady=(0, 10), sticky="w")

    def select_bg(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg")])
        if path:
            self.bg_var.set(path)

    def apply_settings(self):
        self.prompt_entry.insert(0, self.settings.get("prompt", ""))
        self.neg_prompt_entry.insert(0, self.settings.get("negative_prompt", ""))
        self.camera_entry.set(str(self.settings.get("camera", "0")))

    def toggle_preview(self):
        if self.preview_var.get():
            if not self.video_frame.winfo_ismapped():
                self.video_frame.pack(fill=ctk.BOTH, expand=True, padx=10, pady=10)
        else:
            self.video_frame.pack_forget()

    # ------------------------------------------------------------ preview --
    def update_video_frame(self):
        if getattr(self, "closing", False):
            return

        # Drain, buffer and record whenever the engine is running — these used
        # to sit inside the `preview_var` branch, so hiding the preview panel
        # silently froze an in-progress recording and stopped the replay buffer
        # filling, with no indication anything had stopped.
        if self.process is not None:
            import zmq
            import cv2
            import numpy as np

            img_np = None
            try:
                latest = None
                while True:
                    try:
                        latest = self.zmq_socket.recv(zmq.NOBLOCK)
                    except zmq.Again:
                        break
                if latest:
                    img_np = cv2.imdecode(np.frombuffer(latest, np.uint8), cv2.IMREAD_COLOR)
            except Exception:
                img_np = None

            if img_np is not None:
                if not self.saving:
                    self.frame_buffer.append(img_np.copy())

                if self.is_recording and self.video_writer is not None:
                    try:
                        self.video_writer.append_data(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
                    except Exception as e:
                        # Don't fail silently: a broken writer means the file
                        # being produced is garbage.
                        self.is_recording = False
                        try:
                            self.video_writer.close()
                        except Exception:
                            pass
                        self.video_writer = None
                        self.record_btn.configure(text="🔴 Start Recording",
                                                  fg_color="#d9534f", hover_color="#c9302c")
                        self.log(f"[Engine] Recording stopped — writer error: {e}")

                if self.preview_var.get():
                    try:
                        from PIL import Image, ImageTk
                        # Fit the preview to the panel instead of a hardcoded
                        # 768px, which overflowed smaller windows.
                        avail = min(max(self.video_frame.winfo_width(), 64),
                                    max(self.video_frame.winfo_height(), 64))
                        side = max(256, min(avail - 8, 900))
                        shown = cv2.resize(img_np, (side, side), interpolation=cv2.INTER_AREA)
                        pil_img = Image.fromarray(cv2.cvtColor(shown, cv2.COLOR_BGR2RGB))
                        self.current_frame_image = ImageTk.PhotoImage(image=pil_img)
                        self.video_label.configure(image=self.current_frame_image, text="")
                    except Exception:
                        pass

        self.after(15, self.update_video_frame)

    def apply_prompt(self, quiet=False):
        """Push whatever is in the two text boxes to the running engine."""
        prompt = self.prompt_entry.get().strip()
        negative = self.neg_prompt_entry.get().strip()
        if not prompt:
            self.log("[Prompt] Nothing to apply — the prompt box is empty.")
            return
        if self.process is None:
            if not quiet:
                self.log("[Prompt] Engine is not running; this will be used on START.")
            return
        if self.send_command({"prompt": prompt, "negative_prompt": negative}):
            if not quiet:
                self.log(f"[Prompt] Applied: {prompt}")
        else:
            self.log("[Prompt] Could not reach the engine — is it still starting up?")

    def randomize_prompt(self):
        import random
        styles = [
            "cyberpunk neon city, highly detailed, vivid colors",
            "studio ghibli style, lush nature, watercolor, beautiful",
            "grimdark fantasy, gothic, bloodborne, masterpiece",
            "synthwave retrowave 80s, glowing grids",
            "oil painting, classical portrait, rembrandt lighting",
            "wizard with a castle background, fantasy",
            "steampunk inventor workshop, gears, copper",
            "space astronaut on an alien planet, glowing flora",
            "pixar 3D animation style, cute, smooth, rendered in unreal engine",
            "vaporwave aesthetics, pastel colors, glitch art, palm trees",
            "pencil sketch, highly detailed, black and white, hatching",
            "origami paper craft world, colorful, macro photography",
            "holographic neon glowing avatar, futuristic, sci-fi matrix",
            "claymation stop motion style, plasticine, highly detailed",
            "stained glass window portrait, colorful, divine light",
            "retro 90s anime, vhs aesthetic, cel shading"
        ]
        chosen = random.choice(styles)
        self.prompt_entry.delete(0, 'end')
        self.prompt_entry.insert(0, chosen)
        self.log(f"[Randomizer] Rolled style: {chosen}")

        # Same path as typing a prompt and pressing Enter.
        self.apply_prompt(quiet=True)
            
    def take_snapshot(self):
        if len(self.frame_buffer) == 0:
            self.log("[Engine] No frame available to snapshot yet.")
            return
            
        import cv2
        from datetime import datetime

        out_dir = os.path.join(SCRIPT_DIR, "snapshots")
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(out_dir, f"snapshot_{ts}.png")

        # The newest frame off the preview stream. Note this is the engine's
        # 512x512 output after JPEG transport, not a separate high-resolution
        # render — the button used to claim otherwise.
        frame = self.frame_buffer[-1]
        cv2.imwrite(filename, frame)
        self.log(f"[Snapshot] Saved {frame.shape[1]}x{frame.shape[0]} snapshot to {filename}")
        
    def toggle_recording(self):
        import os
        import imageio
        from datetime import datetime
        
        if not self.is_recording:
            # Start Recording
            out_dir = os.path.join(SCRIPT_DIR, "snapshots")
            os.makedirs(out_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(out_dir, f"recording_{ts}.mp4")
            
            try:
                # Use imageio to write universally compatible H.264 MP4s for Discord
                self.video_writer = imageio.get_writer(filename, fps=30.0, codec='libx264', format='FFMPEG')
            except Exception as e:
                self.log(f"[Engine] Failed to open VideoWriter! {e}")
                return
                
            self.is_recording = True
            self.record_btn.configure(text="⏹️ Stop Recording", fg_color="#5bc0de", hover_color="#31b0d5")
            self.log(f"[Engine] Started recording to {filename}...")
        else:
            # Stop Recording
            self.is_recording = False
            if self.video_writer is not None:
                self.video_writer.close()
                self.video_writer = None
            self.record_btn.configure(text="🔴 Start Recording", fg_color="#d9534f", hover_color="#c9302c")
            self.log("[Engine] Recording saved successfully!")

    def save_replay(self):
        if self.saving or len(self.frame_buffer) == 0:
            return
        # Claim the flag on the GUI thread. Setting it inside the worker left a
        # window where two quick Ctrl+S presses started two writers on the same
        # buffer.
        self.saving = True

        import cv2
        import imageio
        from datetime import datetime

        def _do_save():
            try:
                frames = list(self.frame_buffer)
                self.after(0, self.log, f"[Replay] Saving {len(frames)} frames to disk...")

                out_dir = os.path.join(SCRIPT_DIR, "snapshots")
                os.makedirs(out_dir, exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(out_dir, f"replay_{ts}.mp4")

                writer = imageio.get_writer(filename, fps=30.0, codec='libx264', format='FFMPEG')
                for f in frames:
                    writer.append_data(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
                writer.close()
                self.after(0, self.log, f"[Replay] Saved replay to {filename}")
            except Exception as e:
                self.after(0, self.log, f"[Replay] Error saving replay: {e}")
            finally:
                # Always release the flag, otherwise one failed save stops the
                # replay buffer refilling for the rest of the session.
                self.saving = False

        threading.Thread(target=_do_save, daemon=True).start()

    # -------------------------------------------------------------- logic --
    def log(self, message):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", message + "\n")
        
        # Prevent the text box from growing infinitely and lagging the GUI
        lines = int(self.log_box.index('end-1c').split('.')[0])
        if lines > 500:
            self.log_box.delete("1.0", f"{lines - 500}.0")
            
        self.log_box.see("end")
        self.log_box.configure(state="disabled")
        if self.log_file:
            try:
                self.log_file.write(message + "\n")
            except Exception:
                pass

    def read_output(self, pipe):
        try:
            for line in iter(pipe.readline, ''):
                if line:
                    self.after(0, self.log, line.rstrip())
        except Exception:
            pass
        finally:
            try:
                pipe.close()
            except Exception:
                pass

    def python_executable(self):
        venv_py = os.path.join(SCRIPT_DIR, "venv", "Scripts", "python.exe")
        if os.path.exists(venv_py):
            return venv_py
        venv_py = os.path.join(SCRIPT_DIR, "venv", "bin", "python")
        if os.path.exists(venv_py):
            return venv_py
        self.log("venv not found — falling back to the interpreter running this GUI.")
        return sys.executable

    def build_command(self, zmq_port, cmd_port):
        lora_val = self.lora_var.get()
        if lora_val == "None (Original Default)":
            lora_val = "None"
        cmd = [
            self.python_executable(), "-u",
            os.path.join(SCRIPT_DIR, "realtime_video.py"),
            "--prompt", self.prompt_entry.get(),
            "--negative_prompt", self.neg_prompt_entry.get(),
            "--camera", self.camera_entry.get().strip() or "0",
            "--lora", lora_val,
            "--guidance_scale", f"{max(1.05, self.guidance_var.get()):.3f}",
            "--t_index", str(strength_to_t_index(self.strength_var.get())),
            "--freeze_threshold", f"{self.freeze_var.get():.3f}",
            "--motion_smoothing", f"{self.motion_var.get():.2f}",
            "--bokeh_blur", f"{self.bokeh_var.get():.2f}",
            "--expr_overrides", json.dumps({e: v.get() for e, v in self.expr_vars.items()}),
            "--sens_overrides", json.dumps({k: v.get() for k, v in self.sens_vars.items()}),
            "--zmq_port", str(zmq_port),
            "--cmd_port", str(cmd_port)
        ]
        if self.mirror_var.get():
            cmd.append("--mirror_camera")
        if self.vcam_var.get():
            cmd.append("--virtual_camera")
        if self.audio_var.get():
            cmd.append("--audio_sync")
        if self.bg_keep_var.get():
            cmd.append("--composite")
        if self.clahe_var.get():
            cmd.append("--normalize_lighting")
        if self.cudagraph_var.get():
            cmd.append("--cuda_graph")
        if self.bg_var.get().strip():
            cmd += ["--bg_image", self.bg_var.get().strip()]
        return cmd

    def _set_ui_state(self, state):
        self.camera_entry.configure(state=state)
        # self.lora_dropdown.configure(state=state) # Kept active for hot-swapping
        self.preview_cb.configure(state=state)
        self.mirror_cb.configure(state=state)
        self.vcam_cb.configure(state=state)
        self.audio_cb.configure(state=state)
        self.bg_keep_cb.configure(state=state)
        self.bg_entry.configure(state=state)
        self.bg_btn.configure(state=state)
        self.bg_clear_btn.configure(state=state)
        self.clahe_cb.configure(state=state)
        self.cudagraph_cb.configure(state=state)

    def start_script(self):
        if self.process is not None and self.process.poll() is None:
            return

        self.save_settings()
        self.log("=== Booting AI VTuber Engine ===")
        self.log("First run builds TensorRT engines and can take 5-15 minutes.")
        self.video_label.configure(image="", text="Starting engine...")
        self.current_frame_image = None

        # Always wire up the preview socket, even when the panel is hidden.
        # Deciding this from preview_var at launch meant toggling the switch on
        # mid-run left the SUB socket pointing at a dead port from a previous
        # run, and the preview stayed blank until the next full restart.
        import socket
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        zmq_port = sock.getsockname()[1]
        sock.close()
        
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        cmd_port = sock.getsockname()[1]
        sock.close()

        if getattr(self, "last_zmq_port", None):
            try:
                self.zmq_socket.disconnect(f"tcp://127.0.0.1:{self.last_zmq_port}")
            except Exception:
                pass
        if getattr(self, "cmd_socket", None):
            try:
                self.cmd_socket.close()
            except Exception:
                pass
                
        self.last_zmq_port = zmq_port
        self.zmq_socket.connect(f"tcp://127.0.0.1:{zmq_port}")
        
        import zmq
        self.cmd_socket = self.zmq_context.socket(zmq.PUSH)
        # LINGER 0 and an explicit close on shutdown: an open socket blocks
        # context teardown indefinitely.
        self.cmd_socket.setsockopt(zmq.LINGER, 0)
        self.cmd_socket.setsockopt(zmq.SNDTIMEO, 0)
        self.cmd_socket.connect(f"tcp://127.0.0.1:{cmd_port}")

        cmd = self.build_command(zmq_port, cmd_port)
        creationflags = 0
        if sys.platform == "win32":
            # No console window. Shutdown goes over the ZMQ command channel,
            # not console control events — those cannot reach a process that
            # has no console.
            creationflags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP

        try:
            self.process = subprocess.Popen(
                cmd,
                cwd=SCRIPT_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creationflags,
            )
        except Exception as e:
            # Previously a bad interpreter path raised here and the GUI showed
            # nothing at all.
            self.process = None
            self.log(f"[FATAL] Could not start the engine: {e}")
            self.log(f"        command: {' '.join(cmd[:3])} ...")
            self.video_label.configure(text="Engine failed to start.")
            return

        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self._set_ui_state("disabled")
        threading.Thread(target=self.read_output, args=(self.process.stdout,), daemon=True).start()
        threading.Thread(target=self.monitor_process, args=(self.process,), daemon=True).start()

    def monitor_process(self, process):
        code = process.wait()
        self.after(0, self.on_process_exit, code)

    def on_process_exit(self, code):
        if code:
            self.log(f"=== Engine exited with code {code} (see the error above) ===")
        else:
            self.log("=== Engine Shut Down ===")
        # Close the command socket for the run that just ended, so sockets do
        # not accumulate across start/stop cycles and cannot wedge teardown.
        if getattr(self, "cmd_socket", None) is not None:
            try:
                self.cmd_socket.close(linger=0)
            except Exception:
                pass
            self.cmd_socket = None
        if getattr(self, "closing", False):
            return
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self._set_ui_state("normal")
        self.video_label.configure(image="", text="Engine stopped.")
        self.current_frame_image = None
        self.process = None

    def send_command(self, payload):
        """Fire-and-forget a JSON command at the engine. NOBLOCK so a dead or
        unreachable engine can never stall the Tk main loop."""
        import json
        import zmq
        sock = getattr(self, "cmd_socket", None)
        if sock is None:
            return False
        try:
            sock.send_string(json.dumps(payload), flags=zmq.NOBLOCK)
            return True
        except Exception:
            return False

    def stop_script(self):
        if self.process is None:
            return
        self.log("Shutting down engine...")
        self.stop_btn.configure(state="disabled")
        proc = self.process
        # Send from the GUI thread: it is non-blocking and gets the request in
        # before the worker thread starts its wait.
        self.send_command({"cmd": "stop"})
        threading.Thread(target=self.graceful_stop, args=(proc, None), daemon=True).start()

    @staticmethod
    def graceful_stop(proc, _unused=None):
        """Ask over the command channel first, then insist.

        This used to send CTRL_BREAK_EVENT, which cannot be delivered to a
        process created with CREATE_NO_WINDOW — no console is attached. The
        signal never arrived, so every stop sat through the full timeout and
        then hit TerminateProcess, which exits with code 1 and made the GUI
        report a phantom error on every single shutdown. The caller has already
        sent {"cmd": "stop"} over ZMQ, which needs no console at all.
        """
        try:
            proc.wait(timeout=6)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def on_closing(self):
        # Re-entrancy guard: the window stays up while we wait for the engine,
        # so the user can (and will) click X again.
        if getattr(self, "closing", False):
            return
        self.closing = True

        try:
            self.save_settings()
        except Exception:
            pass

        # imageio writers have close(), not release(). The AttributeError was
        # swallowed and the in-progress MP4 was never finalised, so quitting
        # mid-recording left a corrupt file.
        self.is_recording = False
        if getattr(self, "video_writer", None) is not None:
            try:
                self.video_writer.close()
            except Exception:
                pass
            self.video_writer = None

        if self.process is not None:
            self.log("Closing — shutting the engine down...")
            self.send_command({"cmd": "stop"})
            threading.Thread(target=self.graceful_stop,
                             args=(self.process, None), daemon=True).start()
            self._finish_closing(deadline=time.time() + 8.0)
        else:
            self._finish_closing(deadline=0)

    def _finish_closing(self, deadline):
        """Wait for the engine without freezing the window, then tear down."""
        proc = self.process
        if proc is not None and proc.poll() is None and time.time() < deadline:
            self.after(100, self._finish_closing, deadline)
            return

        for sock in (getattr(self, "cmd_socket", None), self.zmq_socket):
            if sock is not None:
                try:
                    sock.close(linger=0)
                except Exception:
                    pass
        self.cmd_socket = None
        try:
            # destroy() closes any socket still open in the context and then
            # terminates it. Plain term() blocks forever on an unclosed socket,
            # and cmd_socket was never being closed — which is why the window
            # refused to go away once the engine had been started.
            self.zmq_context.destroy(linger=0)
        except Exception:
            pass

        if self.log_file:
            try:
                self.log_file.close()
            except Exception:
                pass
        self.destroy()


if __name__ == "__main__":
    try:
        app = VTuberStudioApp()
        app.mainloop()
    except Exception:
        import traceback
        try:
            os.makedirs(LOG_DIR, exist_ok=True)
            with open(os.path.join(LOG_DIR, "launcher-crash.log"), "w", encoding="utf-8") as f:
                traceback.print_exc(file=f)
        except Exception:
            pass
        traceback.print_exc()
        raise
