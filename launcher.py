import json
import os
import subprocess
import sys
import threading
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
        self.frame_buffer = collections.deque(maxlen=50)

        # Global Keybinds
        self.bind("<Control-s>", lambda e: self.save_replay())
        self.bind("<Control-r>", lambda e: self.randomize_prompt())

        # Threading for non-blocking save
        self.saving = False
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
        }
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

        # --- LEFT: prompting + preview ---
        ctk.CTkLabel(self.left_col, text="Master Prompt:", anchor="w").pack(
            fill=ctk.X, padx=10, pady=(10, 0))
        self.prompt_entry = ctk.CTkEntry(self.left_col, placeholder_text="Describe your VTuber...")
        self.prompt_entry.pack(fill=ctk.X, padx=10, pady=5)

        ctk.CTkLabel(self.left_col, text="Negative Prompt:", anchor="w").pack(
            fill=ctk.X, padx=10, pady=(10, 0))
        self.neg_prompt_entry = ctk.CTkEntry(self.left_col)
        self.neg_prompt_entry.pack(fill=ctk.X, padx=10, pady=5)

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
        self.camera_entry = ctk.CTkEntry(self.settings_frame, width=50)
        self.camera_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        ctk.CTkLabel(self.settings_frame, text="Character (LoRA):", anchor="w").grid(
            row=1, column=0, padx=5, pady=5, sticky="w")
        lora_dir = os.path.join(SCRIPT_DIR, "loras")
        loras = ["None"]
        if os.path.isdir(lora_dir):
            loras += sorted(f for f in os.listdir(lora_dir) if f.endswith(".safetensors"))
        saved_lora = self.settings.get("lora", "None")
        self.lora_var = ctk.StringVar(value=saved_lora if saved_lora in loras else "None")
        self.lora_dropdown = ctk.CTkOptionMenu(self.settings_frame, variable=self.lora_var,
                                               values=loras, width=140)
        self.lora_dropdown.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        # Toggles
        self.preview_var = ctk.BooleanVar(value=self.settings.get("embedded_preview", True))
        ctk.CTkSwitch(self.right_col, text="Embedded Preview", variable=self.preview_var,
                      command=self.toggle_preview).pack(anchor="w", padx=15, pady=4)

        self.mirror_var = ctk.BooleanVar(value=self.settings.get("mirror_camera", True))
        ctk.CTkSwitch(self.right_col, text="Mirror Camera (Flip)",
                      variable=self.mirror_var).pack(anchor="w", padx=15, pady=4)

        self.vcam_var = ctk.BooleanVar(value=self.settings.get("virtual_camera", False))
        ctk.CTkSwitch(self.right_col, text="OBS Virtual Camera",
                      variable=self.vcam_var).pack(anchor="w", padx=15, pady=4)

        self.audio_var = ctk.BooleanVar(value=self.settings.get("audio_sync", False))
        ctk.CTkSwitch(self.right_col, text="Audio Lip-Sync (FFT)",
                      variable=self.audio_var).pack(anchor="w", padx=15, pady=4)

        self.bg_keep_var = ctk.BooleanVar(value=self.settings.get("keep_background", False))
        ctk.CTkSwitch(self.right_col, text="Keep Real Background",
                      variable=self.bg_keep_var).pack(anchor="w", padx=15, pady=4)

        # Background image
        self.bg_frame = ctk.CTkFrame(self.right_col, fg_color="transparent")
        self.bg_frame.pack(fill=ctk.X, padx=10, pady=(5, 0))
        ctk.CTkLabel(self.bg_frame, text="BG:", anchor="w").pack(side=ctk.LEFT, padx=5)
        self.bg_var = ctk.StringVar(value=self.settings.get("bg_image", ""))
        ctk.CTkEntry(self.bg_frame, textvariable=self.bg_var, width=100).pack(
            side=ctk.LEFT, padx=5, fill=ctk.X, expand=True)
        ctk.CTkButton(self.bg_frame, text="...", command=self.select_bg, width=30).pack(
            side=ctk.LEFT, padx=(0, 2))
        ctk.CTkButton(self.bg_frame, text="x", command=lambda: self.bg_var.set(""),
                      width=24, fg_color="#555555", hover_color="#777777").pack(side=ctk.LEFT)

        # --- Advanced tuning ---
        ctk.CTkLabel(self.right_col, text="Advanced Tuning:", anchor="w",
                     font=ctk.CTkFont(size=14, weight="bold")).pack(
            fill=ctk.X, padx=10, pady=(15, 5))

        self.tune_frame = ctk.CTkFrame(self.right_col, fg_color="transparent")
        self.tune_frame.pack(fill=ctk.X, padx=10)
        self.tune_frame.grid_columnconfigure(1, weight=1)

        self.strength_var = ctk.DoubleVar(value=self.settings.get("ai_strength", 40.0))
        self._slider_row(0, "AI Strength", self.strength_var, 0, 100, 20,
                         fmt=lambda v: f"{int(v)} (step {strength_to_t_index(v)})")

        self.guidance_var = ctk.DoubleVar(value=self.settings.get("guidance", 1.4))
        self._slider_row(1, "CFG", self.guidance_var, 1.0, 3.0, 20, fmt=lambda v: f"{v:.2f}")

        self.freeze_var = ctk.DoubleVar(value=self.settings.get("freeze", 1.0))
        self._slider_row(2, "Freeze", self.freeze_var, 0.90, 1.00, 10,
                         fmt=lambda v: "off" if v >= 0.999 else f"{v:.2f}")

        self.clahe_var = ctk.BooleanVar(value=self.settings.get("normalize_lighting", False))
        ctk.CTkSwitch(self.right_col, text="Normalize Lighting (CLAHE)",
                      variable=self.clahe_var).pack(anchor="w", padx=15, pady=(8, 4))

        self.cudagraph_var = ctk.BooleanVar(value=self.settings.get("cuda_graph", True))
        ctk.CTkSwitch(self.right_col, text="CUDA Graph (10% faster)",
                      variable=self.cudagraph_var).pack(anchor="w", padx=15, pady=4)

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
        
        self.replay_btn = ctk.CTkButton(self.right_col, text="📷 Save 5s Replay (Ctrl+S)",
                                        command=self.save_replay)
        self.replay_btn.pack(fill=ctk.X, padx=10, pady=5)

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

    def _slider_row(self, row, label, var, lo, hi, steps, fmt):
        ctk.CTkLabel(self.tune_frame, text=f"{label}:", anchor="w").grid(
            row=row, column=0, padx=2, pady=3, sticky="w")
        value_label = ctk.CTkLabel(self.tune_frame, text=fmt(var.get()), width=90, anchor="e")
        value_label.grid(row=row, column=2, padx=2, pady=3, sticky="e")
        slider = ctk.CTkSlider(self.tune_frame, variable=var, from_=lo, to=hi,
                               number_of_steps=steps, width=110,
                               command=lambda v, lbl=value_label, f=fmt: lbl.configure(text=f(v)))
        slider.grid(row=row, column=1, padx=2, pady=3, sticky="ew")

    def select_bg(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg")])
        if path:
            self.bg_var.set(path)

    def apply_settings(self):
        self.prompt_entry.insert(0, self.settings.get("prompt", ""))
        self.neg_prompt_entry.insert(0, self.settings.get("negative_prompt", ""))
        self.camera_entry.insert(0, str(self.settings.get("camera", "0")))

    def toggle_preview(self):
        if self.preview_var.get():
            if not self.video_frame.winfo_ismapped():
                self.video_frame.pack(fill=ctk.BOTH, expand=True, padx=10, pady=10)
        else:
            self.video_frame.pack_forget()

    # ------------------------------------------------------------ preview --
    def update_video_frame(self):
        if self.preview_var.get() and self.process is not None:
            import zmq
            import cv2
            import numpy as np
            from PIL import Image, ImageTk

            try:
                latest = None
                while True:
                    try:
                        latest = self.zmq_socket.recv(zmq.NOBLOCK)
                    except zmq.Again:
                        break

                if latest:
                    img_np = cv2.imdecode(np.frombuffer(latest, np.uint8), cv2.IMREAD_COLOR)
                    if img_np is not None:
                        # Append raw 512x512 frame to our buffer for Replays!
                        if not self.saving:
                            self.frame_buffer.append(img_np.copy())
                            
                        # Fit the preview to the panel instead of a hardcoded
                        # 768px, which overflowed smaller windows.
                        avail = min(max(self.video_frame.winfo_width(), 64),
                                    max(self.video_frame.winfo_height(), 64))
                        side = max(256, min(avail - 8, 900))
                        img_np = cv2.resize(img_np, (side, side), interpolation=cv2.INTER_AREA)
                        pil_img = Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
                        self.current_frame_image = ImageTk.PhotoImage(image=pil_img)
                        self.video_label.configure(image=self.current_frame_image, text="")
            except Exception:
                pass

        self.after(15, self.update_video_frame)

    def randomize_prompt(self):
        import random
        import json
        styles = [
            "cyberpunk neon city, highly detailed, vivid colors",
            "studio ghibli style, lush nature, watercolor, beautiful",
            "grimdark fantasy, gothic, bloodborne, masterpiece",
            "synthwave retrowave 80s, glowing grids",
            "oil painting, classical portrait, rembrandt lighting",
            "wizard with a castle background, fantasy",
            "steampunk inventor workshop, gears, copper",
            "space astronaut on an alien planet, glowing flora"
        ]
        chosen = random.choice(styles)
        self.prompt_entry.delete(0, 'end')
        self.prompt_entry.insert(0, chosen)
        self.log(f"[Randomizer] Rolled style: {chosen}")
        
        # Send dynamic prompt to engine if running
        if hasattr(self, "cmd_socket") and self.process is not None:
            cmd = json.dumps({"prompt": chosen})
            self.cmd_socket.send_string(cmd)
            
    def save_replay(self):
        if self.saving or len(self.frame_buffer) == 0:
            return
            
        import threading
        import cv2
        import os
        from datetime import datetime
        
        def _do_save():
            self.saving = True
            frames = list(self.frame_buffer)
            self.log(f"[Replay] Saving {len(frames)} frames to disk...")
            
            os.makedirs("snapshots", exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"snapshots/replay_{ts}.mp4"
            
            h, w, _ = frames[0].shape
            out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), 10.0, (w, h))
            for f in frames:
                out.write(f)
            out.release()
            
            self.log(f"[Replay] Saved 5-second replay to {filename}!")
            self.saving = False
            
        threading.Thread(target=_do_save, daemon=True).start()

    # -------------------------------------------------------------- logic --
    def log(self, message):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", message + "\n")
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
        cmd = [
            self.python_executable(), "-u",
            os.path.join(SCRIPT_DIR, "realtime_video.py"),
            "--prompt", self.prompt_entry.get(),
            "--negative_prompt", self.neg_prompt_entry.get(),
            "--camera", self.camera_entry.get().strip() or "0",
            "--lora", self.lora_var.get(),
            "--guidance_scale", f"{self.guidance_var.get():.3f}",
            "--t_index", str(strength_to_t_index(self.strength_var.get())),
            "--freeze_threshold", f"{self.freeze_var.get():.3f}",
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
        self.cmd_socket.connect(f"tcp://127.0.0.1:{cmd_port}")

        cmd = self.build_command(zmq_port, cmd_port)
        creationflags = 0
        if sys.platform == "win32":
            # NEW_PROCESS_GROUP is what makes CTRL_BREAK_EVENT deliverable, so
            # STOP can shut the engine down gracefully instead of having
            # TerminateProcess kill it with the webcam still held open.
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
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.video_label.configure(image="", text="Engine stopped.")
        self.current_frame_image = None
        self.process = None

    def stop_script(self):
        if self.process is None:
            return
        self.log("Shutting down engine...")
        self.stop_btn.configure(state="disabled")
        proc = self.process
        threading.Thread(target=self.graceful_stop, args=(proc,), daemon=True).start()

    @staticmethod
    def graceful_stop(proc):
        """Ask first, then insist. The polite signal lets the engine release the
        webcam and close the virtual camera; TerminateProcess does not."""
        import signal
        try:
            if sys.platform == "win32":
                proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                proc.terminate()
        except Exception:
            pass
        try:
            proc.wait(timeout=8)
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
        try:
            self.save_settings()
        except Exception:
            pass
        if self.process is not None:
            self.graceful_stop(self.process)
        try:
            self.zmq_socket.close(linger=0)
            self.zmq_context.term()
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
