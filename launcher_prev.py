import customtkinter as ctk
import subprocess
import threading
import sys
import os
import json

# Setup theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

CONFIG_FILE = "vtuber_settings.json"

class VTuberStudioApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Antigravity VTuber Studio")
        self.geometry("1200x900")
        self.minsize(900, 700)
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.process = None
        self.default_config = {
            "prompt": "1girl, masterpiece, ultra-detailed, cinematic lighting, vibrant colors, flat color, anime key visual, simple background",
            "negative_prompt": "blurry, deformed, bad anatomy, pixelated, jpeg artifacts",
            "camera": "0",
            "virtual_camera": False,
            "audio_sync": False,
            "emotion_sync": False
        }
        self.config = self.load_config()

        # Build UI
        self.build_ui()
        self.apply_config()

    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return self.default_config.copy()

    def save_config(self):
        config = {
            "prompt": self.prompt_entry.get(),
            "negative_prompt": self.neg_prompt_entry.get(),
            "camera": self.camera_entry.get(),
            "embedded_preview": self.preview_var.get(),
            "mirror_camera": self.mirror_var.get(),
            "virtual_camera": self.vcam_var.get(),
            "audio_sync": self.audio_var.get(),
            "lora": self.lora_var.get(),
            "bg_image": self.bg_var.get(),
            "guidance": self.guidance_var.get(),
            "delta": self.delta_var.get(),
        # Removed buffer from config
            "freeze": self.freeze_var.get()
        }
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=4)

    def build_ui(self):
        # Header
        self.header = ctk.CTkLabel(self, text="AI VTuber Studio", font=ctk.CTkFont(size=24, weight="bold"))
        self.header.pack(pady=(20, 10))

        # Main Layout: 2 Columns
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.pack(fill=ctk.BOTH, expand=True, padx=20, pady=10)
        
        self.left_col = ctk.CTkFrame(self.main_frame)
        self.left_col.pack(side=ctk.LEFT, fill=ctk.BOTH, expand=True, padx=(0, 10))
        
        self.right_col = ctk.CTkFrame(self.main_frame, width=250)
        self.right_col.pack(side=ctk.RIGHT, fill=ctk.Y)
        self.right_col.pack_propagate(False)

        # --- LEFT COL: Prompting & Video ---
        ctk.CTkLabel(self.left_col, text="Master Prompt:", anchor="w").pack(fill=ctk.X, padx=10, pady=(10, 0))
        self.prompt_entry = ctk.CTkEntry(self.left_col, placeholder_text="Describe your VTuber...")
        self.prompt_entry.pack(fill=ctk.X, padx=10, pady=5)

        ctk.CTkLabel(self.left_col, text="Negative Prompt:", anchor="w").pack(fill=ctk.X, padx=10, pady=(10, 0))
        self.neg_prompt_entry = ctk.CTkEntry(self.left_col)
        self.neg_prompt_entry.pack(fill=ctk.X, padx=10, pady=5)

        # Video Preview Window
        self.video_frame = ctk.CTkFrame(self.left_col, fg_color="black")
        self.video_frame.pack(fill=ctk.BOTH, expand=True, padx=10, pady=10)
        import tkinter as tk
        self.video_label = tk.Label(self.video_frame, text="Starting Engine...", bg="black", fg="white", font=("Arial", 12))
        self.video_label.pack(expand=True, fill=tk.BOTH)

        # --- RIGHT COL: Settings & Logs ---
        ctk.CTkLabel(self.right_col, text="Studio Settings", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)

        # Hardware & Character
        self.settings_frame = ctk.CTkFrame(self.right_col, fg_color="transparent")
        self.settings_frame.pack(fill=ctk.X, padx=10)
        
        ctk.CTkLabel(self.settings_frame, text="Camera Index:", anchor="w").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.camera_entry = ctk.CTkEntry(self.settings_frame, width=50)
        self.camera_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        ctk.CTkLabel(self.settings_frame, text="Character (LoRA):", anchor="w").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        loras = ["None"]
        if os.path.exists("loras"):
            loras += [f for f in os.listdir("loras") if f.endswith(".safetensors")]
        self.lora_var = ctk.StringVar(value=self.config.get("lora", "None"))
        self.lora_dropdown = ctk.CTkOptionMenu(self.settings_frame, variable=self.lora_var, values=loras, width=120)
        self.lora_dropdown.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        # Toggles
        self.preview_var = ctk.BooleanVar(value=self.config.get("embedded_preview", True))
        self.preview_check = ctk.CTkSwitch(self.right_col, text="Embedded Preview", variable=self.preview_var, command=self.toggle_preview)
        self.preview_check.pack(anchor="w", padx=15, pady=5)

        self.mirror_var = ctk.BooleanVar(value=self.config.get("mirror_camera", True))
        self.mirror_check = ctk.CTkSwitch(self.right_col, text="Mirror Camera (Flip)", variable=self.mirror_var)
        self.mirror_check.pack(anchor="w", padx=15, pady=5)

        self.vcam_var = ctk.BooleanVar(value=self.config.get("virtual_camera", False))
        self.vcam_check = ctk.CTkSwitch(self.right_col, text="OBS Virtual Camera", variable=self.vcam_var)
        self.vcam_check.pack(anchor="w", padx=15, pady=5)

        self.audio_var = ctk.BooleanVar(value=self.config.get("audio_sync", False))
        self.audio_check = ctk.CTkSwitch(self.right_col, text="Audio Lip-Sync (FFT)", variable=self.audio_var)
        self.audio_check.pack(anchor="w", padx=15, pady=5)

        # Chroma Key Background
        self.bg_frame = ctk.CTkFrame(self.right_col, fg_color="transparent")
        self.bg_frame.pack(fill=ctk.X, padx=10, pady=(5,0))
        ctk.CTkLabel(self.bg_frame, text="Background:", anchor="w").pack(side=ctk.LEFT, padx=5)
        self.bg_var = ctk.StringVar(value=self.config.get("bg_image", ""))
        self.bg_entry = ctk.CTkEntry(self.bg_frame, textvariable=self.bg_var, width=100)
        self.bg_entry.pack(side=ctk.LEFT, padx=5, fill=ctk.X, expand=True)
        def select_bg():
            from customtkinter import filedialog
            path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg")])
            if path:
                self.bg_var.set(path)
        self.bg_btn = ctk.CTkButton(self.bg_frame, text="...", command=select_bg, width=30)
        self.bg_btn.pack(side=ctk.LEFT, padx=5)

        # --- Advanced Engine Tuning ---
        ctk.CTkLabel(self.right_col, text="Advanced Tuning:", anchor="w", font=ctk.CTkFont(size=14, weight="bold")).pack(fill=ctk.X, padx=10, pady=(15, 5))
        
        self.tune_frame = ctk.CTkFrame(self.right_col, fg_color="transparent")
        self.tune_frame.pack(fill=ctk.X, padx=10)

        ctk.CTkLabel(self.tune_frame, text="CFG (1-3):", anchor="w").grid(row=0, column=0, padx=2, pady=2, sticky="w")
        self.guidance_var = ctk.DoubleVar(value=self.config.get("guidance", 1.2))
        ctk.CTkSlider(self.tune_frame, variable=self.guidance_var, from_=1.0, to=3.0, number_of_steps=20, width=120).grid(row=0, column=1, padx=2, pady=2)

        ctk.CTkLabel(self.tune_frame, text="Delta (0.5-1.5):", anchor="w").grid(row=1, column=0, padx=2, pady=2, sticky="w")
        self.delta_var = ctk.DoubleVar(value=self.config.get("delta", 1.0))
        ctk.CTkSlider(self.tune_frame, variable=self.delta_var, from_=0.5, to=1.5, number_of_steps=20, width=120).grid(row=1, column=1, padx=2, pady=2)

        # Removed Frame Buffer slider (must mathematically be 1 for 1-Step LCM)

        ctk.CTkLabel(self.tune_frame, text="Freeze (0.9-1.0):", anchor="w").grid(row=3, column=0, padx=2, pady=2, sticky="w")
        self.freeze_var = ctk.DoubleVar(value=self.config.get("freeze", 0.95))
        ctk.CTkSlider(self.tune_frame, variable=self.freeze_var, from_=0.90, to=0.99, number_of_steps=9, width=120).grid(row=3, column=1, padx=2, pady=2)

        # Log Window
        ctk.CTkLabel(self.right_col, text="Engine Logs:", anchor="w").pack(fill=ctk.X, padx=10, pady=(10, 0))
        self.log_box = ctk.CTkTextbox(self.right_col, state="disabled", wrap="word", fg_color="#1E1E1E", height=80)
        self.log_box.pack(fill=ctk.BOTH, expand=True, padx=10, pady=(2, 5))

        # Initialize ZMQ Subscriber
        import zmq
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.SUB)
        self.zmq_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.zmq_socket.RCVTIMEO = 10 # 10ms timeout

        self.toggle_preview()
        self.update_video_frame()

        # Control Buttons
        self.start_btn = ctk.CTkButton(self.right_col, text="▶ START ENGINE", fg_color="#28a745", hover_color="#218838", command=self.start_script)
        self.start_btn.pack(fill=ctk.X, padx=10, pady=(40, 10))

        self.stop_btn = ctk.CTkButton(self.right_col, text="■ STOP", fg_color="#dc3545", hover_color="#c82333", state="disabled", command=self.stop_script)
        self.stop_btn.pack(fill=ctk.X, padx=10, pady=10)

    def apply_config(self):
        self.prompt_entry.insert(0, self.config.get("prompt", ""))
        self.neg_prompt_entry.insert(0, self.config.get("negative_prompt", ""))
        self.camera_entry.insert(0, self.config.get("camera", "0"))
        
    def toggle_preview(self):
        if self.preview_var.get():
            self.video_frame.pack(fill=ctk.BOTH, expand=True, padx=10, pady=(15, 10))
        else:
            self.video_frame.pack_forget()

    def update_video_frame(self):
        if self.preview_var.get() and self.process is not None:
            import zmq
            import cv2
            import numpy as np
            from PIL import Image
            
            try:
                # Try to read all pending frames, keep only the latest
                latest_frame = None
                while True:
                    try:
                        latest_frame = self.zmq_socket.recv(zmq.NOBLOCK)
                    except zmq.Again:
                        break
                
                if latest_frame:
                    nparr = np.frombuffer(latest_frame, np.uint8)
                    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if img_np is not None:
                        # Resize to fit the UI better
                        img_np = cv2.resize(img_np, (768, 768))
                        
                        # Convert to PIL Image for standard Tkinter Label
                        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(img_rgb)
                        
                        from PIL import ImageTk
                        tk_img = ImageTk.PhotoImage(image=pil_img)
                        self.current_frame_image = tk_img # Prevent garbage collection
                        self.video_label.configure(image=self.current_frame_image, text="")
            except Exception as e:
                pass
                
        self.after(10, self.update_video_frame)

    def log(self, message):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", message + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def read_output(self, pipe):
        try:
            for line in iter(pipe.readline, ''):
                if line:
                    self.after(0, self.log, line.strip())
        except Exception:
            pass
        finally:
            pipe.close()

    def start_script(self):
        if self.process is not None and self.process.poll() is None:
            return

        self.save_config() # Save settings when starting

        self.log("=== Booting AI VTuber Engine ===")
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        
        # Determine an open ZMQ port
        import socket
        sock = socket.socket()
        sock.bind(('', 0))
        zmq_port = sock.getsockname()[1]
        sock.close()
        
        if hasattr(self, 'last_zmq_port'):
            try:
                self.zmq_socket.disconnect(f"tcp://127.0.0.1:{self.last_zmq_port}")
            except Exception:
                pass
        self.last_zmq_port = zmq_port
        self.zmq_socket.connect(f"tcp://127.0.0.1:{zmq_port}")

        python_executable = os.path.join(os.getcwd(), "venv", "Scripts", "python.exe")
        
        cmd = [
            python_executable, "-u", "realtime_video.py",
            "--prompt", self.prompt_entry.get(),
            "--negative_prompt", self.neg_prompt_entry.get(),
            "--camera", self.camera_entry.get(),
            "--lora", self.lora_var.get(),
            "--guidance_scale", str(self.guidance_var.get()),
            "--delta", str(self.delta_var.get()),
        # frame_buffer is hardcoded in backend now
            "--freeze_threshold", str(self.freeze_var.get())
        ]
        
        if self.preview_var.get():
            cmd.extend(["--zmq_port", str(zmq_port)])
        if self.mirror_var.get():
            cmd.append("--mirror_camera")
        if self.vcam_var.get():
            cmd.append("--virtual_camera")
        if self.audio_var.get():
            cmd.append("--audio_sync")
        if self.bg_var.get():
            cmd.append("--bg_image")
            cmd.append(self.bg_var.get())

        creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            bufsize=1,
            creationflags=creationflags
        )

        threading.Thread(target=self.read_output, args=(self.process.stdout,), daemon=True).start()
        threading.Thread(target=self.monitor_process, daemon=True).start()

    def monitor_process(self):
        self.process.wait()
        self.after(0, self.on_process_exit)

    def on_process_exit(self):
        self.log("=== Engine Shut Down ===")
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.process = None

    def stop_script(self):
        if self.process is not None:
            self.log("Shutting down engine...")
            self.process.terminate()

    def on_closing(self):
        self.save_config()
        if self.process is not None:
            self.process.terminate()
        self.destroy()

if __name__ == "__main__":
    app = VTuberStudioApp()
    app.mainloop()
