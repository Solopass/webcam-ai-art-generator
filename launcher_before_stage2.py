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

# Presets and history live in their OWN files, deliberately. save_settings()
# rebuilds vtuber_settings.json from the widgets on every START and on close,
# so anything stored in there that is not backed by a widget gets erased —
# which is exactly what happens to the 💾 prompt list today.
PRESETS_FILE = os.path.join(SCRIPT_DIR, "presets.json")
HISTORY_FILE = os.path.join(SCRIPT_DIR, "history.json")
HISTORY_LIMIT = 50

# Settings the engine reads once at launch. Everything else has a live handler
# in cmd_listener_thread and can be pushed over ZMQ mid-run.
RESTART_ONLY_KEYS = {
    "camera", "perf_mode", "controlnet", "ai_strength", "cuda_graph",
    "virtual_camera", "audio_sync", "no_face_track", "normalize_lighting",
    "bg_image",
}

# Seeded style presets. These are PARTIAL: they carry style-relevant keys only,
# so loading one never touches your camera index, OBS toggle or preview setup.
BUILTIN_PRESETS = {
    "★ Flat 2D Anime": {
        "prompt": "1girl, anime screencap, flat color, cel shading, bold lineart, "
                  "simple background, masterpiece",
        "negative_prompt": "3d, render, realistic, photo, photorealistic, octane, "
                           "blender, depth of field, film grain, blurry, deformed",
        "ai_strength": 82.0,          # t_index 18 — enough noise to actually repaint
        "guidance": 1.6,
        "normalize_lighting": False,  # CLAHE strengthens the real 3D shading
        "freeze": 1.0,                # off: the stillness pass blends the webcam back in
        "motion_smoothing": 0.4,
        "bokeh_blur": 0.0,
        "easynegative": True,
        "keep_background": False,
        "controlnet": "None",
        "zoom": 1.0,
        "perf_mode": "Standard (2-Step)",
    },
    "★ Full Frame Anime": {
        # Flat 2D, but painting the ENTIRE frame instead of a face-tracked
        # square cut to your silhouette. Needs all three of zoom, face tracking
        # and segmentation off — leave any one on and the AI is confined again.
        "prompt": "anime screencap, flat color, cel shading, bold lineart, "
                  "detailed background, masterpiece",
        "negative_prompt": "3d, render, realistic, photo, photorealistic, octane, "
                           "blender, depth of field, film grain, blurry, deformed",
        "ai_strength": 82.0,
        "guidance": 1.6,
        "normalize_lighting": False,
        "freeze": 1.0,
        "motion_smoothing": 0.4,
        "bokeh_blur": 0.0,
        "easynegative": True,
        "zoom": 1.0,
        "no_face_track": True,      # whole frame, not a square
        "no_segment": True,         # no grey-screen in, no silhouette mask out
        "keep_background": False,
        "controlnet": "None",
        "perf_mode": "Standard (2-Step)",
    },
    "★ Painterly": {
        "prompt": "oil painting, thick brush strokes, impasto, painterly portrait, "
                  "dramatic lighting, masterpiece",
        "negative_prompt": "3d, render, photo, blurry, deformed, text, watermark",
        "ai_strength": 65.0,          # t_index 24
        "guidance": 1.8,
        "normalize_lighting": True,
        "freeze": 0.99,
        "motion_smoothing": 0.6,
        "bokeh_blur": 0.0,
        "easynegative": True,
        "zoom": 1.0,
        "perf_mode": "Standard (2-Step)",
    },
    "★ Subtle Filter": {
        "prompt": "soft anime style, gentle shading, clean lines, natural colors",
        "negative_prompt": "blurry, deformed, extra limbs, text, watermark",
        "ai_strength": 30.0,          # t_index 35 — your real face, lightly stylised
        "guidance": 1.4,
        "normalize_lighting": False,
        "freeze": 0.99,
        "motion_smoothing": 0.7,
        "bokeh_blur": 0.0,
        "easynegative": True,
        "zoom": 1.0,
        "perf_mode": "Standard (2-Step)",
    },
}

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
    # Image controls. These defaults reproduce values that used to be
    # hardcoded in the engine, so the picture is unchanged out of the box.
    "sharpness": 1.0,
    "saturation": 20.0,
    "brightness": 10.0,
    "mask_feather": 7.0,
    "temporal_denoise": 0.4,
    "stillness_blend": 0.3,
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
        self.geometry("1400x1000")
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
        self.bind("<Control-space>", lambda e: self.take_snapshot())
        self.bind("<Control-f>", lambda e: self.toggle_freeze())
        self.bind("<F8>", lambda e: self.toggle_freeze())

        # Threading for non-blocking save
        self.saving = False

        # (var, value_label, fmt) for every slider built by _slider_row
        self._slider_rows = []
        self._slider_widgets = {}
        self._history_entries = []
        
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

    def _collect_settings(self):
        """Every persisted setting, read out of the widgets.

        Single source of truth shared by save_settings(), preset saving and the
        history snapshot, so the three can never drift apart.
        """
        settings = {
            "prompt": self.prompt_entry.get(),
            "negative_prompt": self.neg_prompt_entry.get(),
            "camera": self.camera_entry.get(),
            "embedded_preview": self.preview_var.get(),
            "mirror_camera": self.mirror_var.get(),
            "virtual_camera": self.vcam_var.get(),
            "no_face_track": self.no_face_track_var.get(),
            "no_segment": self.no_segment_var.get(),
            "sharpness": self.sharp_var.get(),
            "saturation": self.sat_var.get(),
            "brightness": self.bright_var.get(),
            "mask_feather": self.feather_var.get(),
            "temporal_denoise": self.denoise_var.get(),
            "stillness_blend": self.stillness_var.get(),
            "audio_sync": self.audio_var.get(),
            "keep_background": self.bg_keep_var.get(),
            "normalize_lighting": self.clahe_var.get(),
            "cuda_graph": self.cudagraph_var.get(),
            "lora": self.lora_var.get(),
            "controlnet": self.controlnet_var.get(),
            "perf_mode": self.perf_var.get(),
            "easynegative": self.easyneg_var.get(),
            "bg_image": self.bg_var.get(),
            "guidance": self.guidance_var.get(),
            "ai_strength": self.strength_var.get(),
            "freeze": self.freeze_var.get(),
            "motion_smoothing": self.motion_var.get(),
            "bokeh_blur": self.bokeh_var.get(),
            "zoom": self.zoom_var.get(),
        }
        for e, v in self.expr_vars.items():
            settings[f"expr_{e}"] = v.get()
        for k, v in self.sens_vars.items():
            settings[f"sens_{k}"] = v.get()
        return settings

    def save_settings(self):
        settings = self._collect_settings()
        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(settings, f, indent=4)
        except Exception as e:
            self.log(f"Could not save settings: {e}")

    # ------------------------------------------------ presets & history --
    @staticmethod
    def _read_json(path, fallback):
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, type(fallback)):
                    return data
        except Exception:
            pass
        return fallback

    def _write_json(self, path, data, what):
        try:
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, path)   # atomic: a crash mid-write can't truncate it
            return True
        except Exception as e:
            self.log(f"[{what}] Could not write {os.path.basename(path)}: {e}")
            return False

    def _apply_settings_dict(self, values, push_live=True):
        """Write a (possibly partial) settings dict into the widgets.

        Returns the list of restart-only keys that actually changed, so the
        caller can tell the user which parts will not take effect until the
        next START.
        """
        changed_restart_only = []

        for key, value in values.items():
            try:
                if key == "prompt":
                    if self.prompt_entry.get() != value:
                        self.prompt_entry.delete(0, "end")
                        self.prompt_entry.insert(0, value)
                elif key == "negative_prompt":
                    if self.neg_prompt_entry.get() != value:
                        self.neg_prompt_entry.delete(0, "end")
                        self.neg_prompt_entry.insert(0, value)
                elif key == "camera":
                    if self.camera_entry.get() != str(value):
                        self.camera_entry.set(str(value))
                        changed_restart_only.append(key)
                elif key.startswith("expr_"):
                    name = key[len("expr_"):]
                    if name in self.expr_vars:
                        self.expr_vars[name].set(value)
                elif key.startswith("sens_"):
                    name = key[len("sens_"):]
                    if name in self.sens_vars:
                        self.sens_vars[name].set(value)
                else:
                    var = self._settings_var(key)
                    if var is None:
                        continue
                    if var.get() != value:
                        var.set(value)
                        if key in RESTART_ONLY_KEYS:
                            changed_restart_only.append(key)
            except Exception as e:
                self.log(f"[Preset] Skipped '{key}': {e}")

        self._refresh_slider_labels()
        self.toggle_preview()

        if push_live and self.process is not None:
            self._push_live_settings()

        return changed_restart_only

    def _settings_var(self, key):
        """Map a settings key onto its Tk variable, or None if it has no widget."""
        mapping = {
            "embedded_preview": "preview_var",
            "mirror_camera": "mirror_var",
            "virtual_camera": "vcam_var",
            "no_face_track": "no_face_track_var",
            "no_segment": "no_segment_var",
            "sharpness": "sharp_var",
            "saturation": "sat_var",
            "brightness": "bright_var",
            "mask_feather": "feather_var",
            "temporal_denoise": "denoise_var",
            "stillness_blend": "stillness_var",
            "audio_sync": "audio_var",
            "keep_background": "bg_keep_var",
            "normalize_lighting": "clahe_var",
            "cuda_graph": "cudagraph_var",
            "lora": "lora_var",
            "controlnet": "controlnet_var",
            "perf_mode": "perf_var",
            "easynegative": "easyneg_var",
            "bg_image": "bg_var",
            "guidance": "guidance_var",
            "ai_strength": "strength_var",
            "freeze": "freeze_var",
            "motion_smoothing": "motion_var",
            "bokeh_blur": "bokeh_var",
            "zoom": "zoom_var",
            "vfx_opacity": "vfx_op_var",
            "vfx_blend_mode": "vfx_blend_var",
        }
        return getattr(self, mapping[key], None) if key in mapping else None

    def _send_no_segment(self):
        self.send_command({"no_segment": bool(self.no_segment_var.get())})

    def _push_live_settings(self):
        """Send everything the running engine can accept mid-run."""
        self.apply_prompt(quiet=True)     # handles the EasyNegative suffix too
        for payload in (
            {"guidance_scale": max(1.05, float(self.guidance_var.get()))},
            {"freeze_threshold": float(self.freeze_var.get())},
            {"motion_smoothing": float(self.motion_var.get())},
            {"bokeh_blur": float(self.bokeh_var.get())},
            {"zoom": float(self.zoom_var.get())},
            {"expr_override": {e: v.get() for e, v in self.expr_vars.items()}},
            {"sens_override": {k: float(v.get()) for k, v in self.sens_vars.items()}},
            {"composite": bool(self.bg_keep_var.get())},
            {"no_segment": bool(self.no_segment_var.get())},
            {"sharpness": float(self.sharp_var.get())},
            {"saturation": int(self.sat_var.get())},
            {"brightness": int(self.bright_var.get())},
            {"mask_feather": int(self.feather_var.get())},
            {"temporal_denoise": float(self.denoise_var.get())},
            {"stillness_blend": float(self.stillness_var.get())},
        ):
            self.send_command(payload)

    def _refresh_slider_labels(self):
        """Sliders set programmatically do not fire their command callback, so
        the value labels would keep showing the old numbers."""
        for var, label, fmt in getattr(self, "_slider_rows", []):
            try:
                label.configure(text=fmt(var.get()))
            except Exception:
                pass

    # --- presets ---
    def user_presets(self):
        return self._read_json(PRESETS_FILE, {})

    def preset_names(self):
        return list(BUILTIN_PRESETS.keys()) + sorted(self.user_presets().keys())

    def _refresh_preset_menu(self, select=None):
        names = self.preset_names()
        self.preset_menu.configure(values=names or ["(none saved)"])
        if select and select in names:
            self.preset_var.set(select)

    def on_preset_load(self):
        name = self.preset_var.get()
        values = BUILTIN_PRESETS.get(name) or self.user_presets().get(name)
        if not values:
            self.log(f"[Preset] '{name}' not found.")
            return

        needs_restart = self._apply_settings_dict(values)
        self.log(f"[Preset] Loaded '{name}' ({len(values)} settings).")

        if needs_restart:
            pretty = ", ".join(sorted(needs_restart))
            if self.process is not None:
                self.log(f"[Preset] Applied live where possible. Needs a STOP/START "
                         f"to take effect: {pretty}")
            else:
                self.log(f"[Preset] Launch-only settings staged for next START: {pretty}")

    def on_preset_save(self):
        dialog = ctk.CTkInputDialog(text="Name this preset:", title="Save Preset")
        name = (dialog.get_input() or "").strip()
        if not name:
            return
        if name.startswith("★"):
            self.log("[Preset] '★' is reserved for the built-in presets.")
            return

        presets = self.user_presets()
        existed = name in presets
        presets[name] = self._collect_settings()
        if self._write_json(PRESETS_FILE, presets, "Preset"):
            self._refresh_preset_menu(select=name)
            self.log(f"[Preset] {'Updated' if existed else 'Saved'} '{name}'.")

    def on_preset_delete(self):
        name = self.preset_var.get()
        if name in BUILTIN_PRESETS:
            self.log("[Preset] Built-in presets cannot be deleted.")
            return
        presets = self.user_presets()
        if name not in presets:
            self.log(f"[Preset] '{name}' is not a saved preset.")
            return
        del presets[name]
        if self._write_json(PRESETS_FILE, presets, "Preset"):
            self._refresh_preset_menu()
            names = self.preset_names()
            self.preset_var.set(names[0] if names else "")
            self.log(f"[Preset] Deleted '{name}'.")

    # --- history ---
    def load_history(self):
        return self._read_json(HISTORY_FILE, [])

    def append_history(self):
        """Snapshot the settings every time the engine starts."""
        entry = {
            "at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "settings": self._collect_settings(),
        }
        history = self.load_history()

        # Starting the engine five times to test one prompt should leave one
        # entry, not five.
        if history and history[-1].get("settings") == entry["settings"]:
            return

        history.append(entry)
        if len(history) > HISTORY_LIMIT:
            history = history[-HISTORY_LIMIT:]
        if self._write_json(HISTORY_FILE, history, "History"):
            self._refresh_history_menu()

    @staticmethod
    def _history_label(idx, entry):
        prompt = (entry.get("settings", {}).get("prompt") or "").strip()
        if len(prompt) > 34:
            prompt = prompt[:33] + "…"
        return f"{idx:>2}. {entry.get('at', '?')} — {prompt or '(no prompt)'}"

    def _refresh_history_menu(self):
        history = self.load_history()
        # Newest first: what you want back is almost always what you just had.
        self._history_entries = list(reversed(history))
        labels = [self._history_label(i + 1, e) for i, e in enumerate(self._history_entries)]
        self.history_menu.configure(values=labels or ["(no history yet)"])
        self.history_var.set(labels[0] if labels else "(no history yet)")

    def on_history_restore(self):
        labels = [self._history_label(i + 1, e)
                  for i, e in enumerate(getattr(self, "_history_entries", []))]
        label = self.history_var.get()
        if label not in labels:
            self.log("[History] Nothing to restore yet — start the engine once.")
            return
        entry = self._history_entries[labels.index(label)]
        needs_restart = self._apply_settings_dict(entry.get("settings", {}))
        self.log(f"[History] Restored the settings from {entry.get('at', '?')}.")
        if needs_restart and self.process is not None:
            self.log(f"[History] Needs a STOP/START for: {', '.join(sorted(needs_restart))}")

    # ---------------------------------------------------------------- UI --
    def build_ui(self):
        self.header = ctk.CTkLabel(self, text="AI VTuber Studio",
                                   font=ctk.CTkFont(size=24, weight="bold"))
        self.header.pack(pady=(20, 10))

        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.pack(fill=ctk.BOTH, expand=True, padx=20, pady=10)

        self.left_col = ctk.CTkFrame(self.main_frame)
        self.left_col.pack(side=ctk.LEFT, fill=ctk.BOTH, expand=True, padx=(0, 10))

        self.right_container = ctk.CTkFrame(self.main_frame, width=400)
        self.right_container.pack(side=ctk.RIGHT, fill=ctk.Y)
        
        self.tabview = ctk.CTkTabview(self.right_container, width=400)
        self.tabview.pack(fill=ctk.BOTH, expand=True)
        
        self.tab_settings = self.tabview.add("Settings")
        self.tab_advanced = self.tabview.add("Advanced")
        self.tab_system = self.tabview.add("System")
        self.tab_vfx = self.tabview.add("VFX Pipeline")
        
        self.right_col = ctk.CTkScrollableFrame(self.tab_settings)
        self.right_col.pack(fill=ctk.BOTH, expand=True)
        
        self.advanced_col = ctk.CTkScrollableFrame(self.tab_advanced)
        self.advanced_col.pack(fill=ctk.BOTH, expand=True)
        
        self.system_col = ctk.CTkFrame(self.tab_system)
        self.system_col.pack(fill=ctk.BOTH, expand=True)

        # --- LEFT: prompting + preview ---
        prompt_builder_frame = ctk.CTkFrame(self.left_col, fg_color="transparent")
        prompt_builder_frame.pack(fill=ctk.X, padx=10, pady=(10, 0))
        
        self.prompt_dropdowns = []
        
        def load_prompts_from_md():
            prompts = {}
            current_category = "Uncategorized"
            md_path = os.path.join(SCRIPT_DIR, "prompts.md")
            if not os.path.exists(md_path):
                return {"Uncategorized": []}
            with open(md_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    if line.startswith("# "):
                        current_category = line[2:].strip()
                        if current_category not in prompts:
                            prompts[current_category] = []
                    elif line.startswith("- "):
                        prompts[current_category].append(line[2:].strip())
                    else:
                        prompts[current_category].append(line)
            return prompts

        def build_dropdowns():
            for widget in prompt_builder_frame.winfo_children():
                widget.destroy()
            self.prompt_dropdowns.clear()
            
            categories = load_prompts_from_md()
            row = ctk.CTkFrame(prompt_builder_frame, fg_color="transparent")
            row.pack(fill=ctk.X)
            
            for cat, items in categories.items():
                if not items or cat == "Uncategorized": continue
                val_list = [f"-- {cat} --"] + items
                dd = ctk.CTkOptionMenu(row, values=val_list, width=110, text_color=("black", "white"))
                dd.set(f"-- {cat} --")
                dd.pack(side=ctk.LEFT, padx=2, pady=2)
                self.prompt_dropdowns.append((cat, dd))
                
            def combine_prompts():
                parts = []
                for cat, dd in self.prompt_dropdowns:
                    v = dd.get()
                    if not v.startswith("-- "):
                        parts.append(v)
                if parts:
                    self.prompt_entry.delete(0, 'end')
                    self.prompt_entry.insert(0, ", ".join(parts))
                    self.apply_prompt()
            
            ctk.CTkButton(row, text="Generate", width=70, command=combine_prompts).pack(side=ctk.LEFT, padx=5)
            ctk.CTkButton(row, text="?", width=30, command=build_dropdowns).pack(side=ctk.LEFT)

        build_dropdowns()

        def build_prompt_header(label_text, entry_box, settings_key, header):
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
                    
            def on_delete():
                current = entry_box.get().strip()
                if current in saved:
                    saved.remove(current)
                    self.settings[settings_key] = saved
                    self.save_settings()
                    dropdown.configure(values=[label_text] + saved)
                    entry_box.delete(0, 'end')
                    self.apply_prompt()
                    
            dropdown = ctk.CTkOptionMenu(header, values=[label_text] + saved, command=on_select,
                                         text_color=("black", "white"))
            dropdown.set(label_text)
            dropdown.pack(side=ctk.LEFT)
            ctk.CTkButton(header, text="💾", width=30, height=24, fg_color="transparent",
                          command=on_save, hover_color="#333333").pack(side=ctk.LEFT, padx=(5, 0))
            ctk.CTkButton(header, text="🗑️", width=30, height=24, fg_color="transparent",
                          command=on_delete, hover_color="#333333").pack(side=ctk.LEFT, padx=5)


        header_main = ctk.CTkFrame(self.left_col, fg_color="transparent")
        header_main.pack(fill=ctk.X, padx=10, pady=(10, 0))
        
        prompt_row = ctk.CTkFrame(self.left_col, fg_color="transparent")
        prompt_row.pack(fill=ctk.X, padx=10, pady=5)
        
        self.prompt_entry = ctk.CTkEntry(prompt_row, placeholder_text="Master Prompt...")
        self.prompt_entry.pack(side=ctk.LEFT, fill=ctk.X, expand=True)
        
        build_prompt_header("Saved Prompts 💾", self.prompt_entry, "saved_main_prompts", header_main)
        
        self.apply_btn = ctk.CTkButton(prompt_row, text="Apply ✨", width=86, command=self.apply_prompt)
        self.apply_btn.pack(side=ctk.LEFT, padx=(6, 0))

        header_neg = ctk.CTkFrame(self.left_col, fg_color="transparent")
        header_neg.pack(fill=ctk.X, padx=10, pady=(10, 0))
        self.neg_prompt_entry = ctk.CTkEntry(self.left_col)
        self.neg_prompt_entry.pack(fill=ctk.X, padx=10, pady=5)
        build_prompt_header("Negative Prompt 🚫", self.neg_prompt_entry, "saved_neg_prompts", header_neg)

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

        # --- Style presets ---
        preset_box = ctk.CTkFrame(self.right_col)
        preset_box.pack(fill=ctk.X, padx=10, pady=(0, 10))
        ctk.CTkLabel(preset_box, text="Style Preset", anchor="w",
                     font=ctk.CTkFont(size=13, weight="bold")).pack(
            fill=ctk.X, padx=8, pady=(6, 0))
        ctk.CTkLabel(preset_box,
                     text="★ presets only change style settings — your camera, "
                          "OBS and preview options are left alone.",
                     font=ctk.CTkFont(size=11), text_color="gray",
                     justify="left", wraplength=340, anchor="w").pack(
            fill=ctk.X, padx=8, pady=(0, 4))

        names = self.preset_names()
        self.preset_var = ctk.StringVar(value=names[0] if names else "")
        self.preset_menu = ctk.CTkOptionMenu(preset_box, variable=self.preset_var,
                                             values=names or ["(none saved)"])
        self.preset_menu.pack(fill=ctk.X, padx=8, pady=(0, 6))

        preset_btns = ctk.CTkFrame(preset_box, fg_color="transparent")
        preset_btns.pack(fill=ctk.X, padx=8, pady=(0, 8))
        ctk.CTkButton(preset_btns, text="Load", command=self.on_preset_load,
                      width=70).pack(side=ctk.LEFT, expand=True, fill=ctk.X, padx=(0, 4))
        ctk.CTkButton(preset_btns, text="Save As…", command=self.on_preset_save,
                      width=80).pack(side=ctk.LEFT, expand=True, fill=ctk.X, padx=4)
        ctk.CTkButton(preset_btns, text="Delete", command=self.on_preset_delete,
                      width=70, fg_color="#8a3b3b", hover_color="#a04545").pack(
            side=ctk.LEFT, expand=True, fill=ctk.X, padx=(4, 0))

        self.settings_frame = ctk.CTkFrame(self.right_col, fg_color="transparent")
        self.settings_frame.pack(fill=ctk.X, padx=10)
        self.settings_frame.columnconfigure(1, weight=1)

        ctk.CTkLabel(self.settings_frame, text="Camera Index:", anchor="w").grid(
            row=0, column=0, padx=5, pady=5, sticky="ew")
        self.camera_entry = ctk.CTkComboBox(self.settings_frame, width=150, values=["0", "1", "screen 1 (Primary)", "screen 2 (Secondary)", "screen 3 (Tertiary)"])
        self.camera_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        ctk.CTkLabel(self.settings_frame, text="Character (LoRA):", anchor="w").grid(
            row=1, column=0, padx=5, pady=5, sticky="ew")
        lora_dir = os.path.join(SCRIPT_DIR, "loras")
        loras = ["None (Original Default)"]
        if os.path.isdir(lora_dir):
            loras += sorted(f for f in os.listdir(lora_dir) if f.endswith(".safetensors"))
        def on_lora_changed(val):
            if val == "None (Original Default)":
                val = "None"
            if self.process is None:
                return
            # The refit re-fuses the UNet, re-exports ONNX and reloads TRT
            # weights on the inference thread — 30-60s with the output frozen.
            # Grey the dropdown so that reads as "busy", not "broken".
            if not self.send_command({"lora": val}):
                self.log("[LoRA] Could not reach the engine — is it still starting?")
                return
            self.log(f"[LoRA] Swapping to '{val}'. Output freezes for 30-60s "
                     f"while the engine refits — this is normal.")
            try:
                self.lora_dropdown.configure(state="disabled")
            except Exception:
                pass
            self._lora_swap_pending = True
            # Fallback in case the engine dies mid-swap and never reports back.
            self.after(120000, self._end_lora_swap)

        saved_lora = self.settings.get("lora", "None (Original Default)")
        if saved_lora == "None":
            saved_lora = "None (Original Default)"
        self.lora_var = ctk.StringVar(value=saved_lora if saved_lora in loras else "None (Original Default)")
        self.lora_dropdown = ctk.CTkOptionMenu(self.settings_frame, variable=self.lora_var,
                                               values=loras, width=140, command=on_lora_changed)
        self.lora_dropdown.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        ctk.CTkLabel(self.settings_frame, text="ControlNet Mode:", anchor="w").grid(
            row=2, column=0, padx=5, pady=5, sticky="ew")
        saved_cnet = self.settings.get("controlnet", "None")
        self.controlnet_var = ctk.StringVar(value=saved_cnet)
        self.controlnet_dropdown = ctk.CTkOptionMenu(
            self.settings_frame, variable=self.controlnet_var,
            values=["None", "Depth (MiDaS)", "Canny Edge (Details)", "Lineart (Sketches)", "OpenPose (Skeletal)", "Depth + Canny (Heavy/Low FPS)"], width=140)
        self.controlnet_dropdown.grid(row=2, column=1, padx=5, pady=5, sticky="ew")

        ctk.CTkLabel(self.settings_frame, text="Performance Mode:", anchor="w").grid(
            row=3, column=0, padx=5, pady=5, sticky="ew")
        # "Maximum Speed" and "Balanced" both resolved to fb=1/steps=2 — the
        # same engine, the same work. Collapsed into one honest entry. (A
        # frame_buffer of 2 is NOT a speed mode: the code feeds the same frame
        # twice and discards one output, i.e. double the UNet work for an
        # identical result.) Old names still load via the fallback below.
        perf_modes = [
            "Standard (2-Step)",
            "Maximum Quality (4-Step, Low FPS)"
        ]
        saved_perf = self.settings.get("perf_mode", perf_modes[0])
        # Map the retired names onto their real behaviour so old saved settings,
        # presets and history entries keep working.
        _LEGACY_PERF = {
            "Maximum Speed (Low Quality)": perf_modes[0],
            "Balanced (Recommended for ControlNet)": perf_modes[0],
            "Maximum Quality (Low FPS)": perf_modes[1],
        }
        saved_perf = _LEGACY_PERF.get(saved_perf, saved_perf)
        self.perf_var = ctk.StringVar(value=saved_perf if saved_perf in perf_modes else perf_modes[0])
        self.perf_dropdown = ctk.CTkOptionMenu(self.settings_frame, variable=self.perf_var, values=perf_modes, width=140)
        self.perf_dropdown.grid(row=3, column=1, padx=5, pady=5, sticky="ew")

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

        self.easyneg_var = ctk.BooleanVar(value=self.settings.get("easynegative", True))
        self.easyneg_cb = ctk.CTkSwitch(self.right_col, text="Use EasyNegative Embeds",
                                        command=self.apply_prompt, variable=self.easyneg_var)
        self.easyneg_cb.pack(anchor="w", padx=15, pady=4)

        self.bg_keep_var = ctk.BooleanVar(value=self.settings.get("keep_background", False))
        self.bg_keep_cb = ctk.CTkSwitch(self.right_col, text="Composite Real Background",
                                        command=lambda: self.cmd_socket.send_string(__import__("json").dumps({"composite": self.bg_keep_var.get()})) if getattr(self, "cmd_socket", None) else None, variable=self.bg_keep_var)
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
        ctk.CTkLabel(self.advanced_col, text="Advanced Tuning:", anchor="w",
                     font=ctk.CTkFont(size=14, weight="bold")).pack(
            fill=ctk.X, padx=10, pady=(15, 5))

        self.tune_frame = ctk.CTkFrame(self.advanced_col, fg_color="transparent")
        self.tune_frame.pack(fill=ctk.X, padx=10)
        self.tune_frame.grid_columnconfigure(1, weight=1)

        self.strength_var = ctk.DoubleVar(value=self.settings.get("ai_strength", 40.0))
        self._slider_row(0, "AI Strength", self.strength_var, 0, 100, 20,
                         fmt=lambda v: f"{int(v)} (step {strength_to_t_index(v)})",
                         desc="Higher = closer to raw webcam, Lower = more AI stylization. (Requires Restart)")

        # Floor of 1.05, not 1.0: at exactly 1.0 StreamDiffusion disables
        # classifier-free guidance entirely, which halves the prompt-embed
        # tensor and changes the UNet batch out from under the built engine.
        self.guidance_var = ctk.DoubleVar(value=max(1.05, self.settings.get("guidance", 1.4)))
        self._slider_row(1, "Prompt Strictness (CFG)", self.guidance_var, 1.05, 3.0, 20, fmt=lambda v: f"{v:.2f}",
                         on_change=lambda v: self.cmd_socket.send_string(__import__("json").dumps({"guidance_scale": float(v)})) if getattr(self, "cmd_socket", None) else None,
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

        self.motion_var = ctk.DoubleVar(value=self.settings.get("motion_smoothing", 0.0))
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

        def _send_zoom(v):
            if getattr(self, "cmd_socket", None):
                try: self.cmd_socket.send_string(__import__("json").dumps({"zoom": float(v)}))
                except Exception: pass

        self.zoom_var = ctk.DoubleVar(value=self.settings.get("zoom", 1.0))
        self._slider_row(5, "Camera Zoom", self.zoom_var, 1.0, 3.0, 40,
                         fmt=lambda v: f"{v:.1f}x",
                         on_change=_send_zoom,
                         desc="Zooms the camera in to focus tighter on your face.")



        def _live(key, cast=float):
            return lambda v: self.send_command({key: cast(v)})

        self.sharp_var = ctk.DoubleVar(value=self.settings.get("sharpness", 1.0))
        self._slider_row(6, "Sharpness", self.sharp_var, 0.0, 2.0, 20,
                         fmt=lambda v: f"{v:.2f}", on_change=_live("sharpness"),
                         desc="Crispness of the AI output. 0 = off. Higher re-adds "
                              "photographic micro-detail, which reads as less flat.")

        self.sat_var = ctk.DoubleVar(value=self.settings.get("saturation", 20.0))
        self._slider_row(7, "Saturation", self.sat_var, -40, 60, 20,
                         fmt=lambda v: f"{int(v):+d}", on_change=_live("saturation", int),
                         desc="Colour boost applied after generation.")

        self.bright_var = ctk.DoubleVar(value=self.settings.get("brightness", 10.0))
        self._slider_row(8, "Brightness", self.bright_var, -30, 40, 14,
                         fmt=lambda v: f"{int(v):+d}", on_change=_live("brightness", int),
                         desc="Lift or drop the output value channel.")

        self.feather_var = ctk.DoubleVar(value=self.settings.get("mask_feather", 7.0))
        self._slider_row(9, "Mask Feather", self.feather_var, 1, 31, 15,
                         fmt=lambda v: f"{int(v) | 1}px", on_change=_live("mask_feather", int),
                         desc="Softness of the cutout edge when compositing. Higher "
                              "hides a hard pasted-on edge. No effect with Paint "
                              "Whole Frame on.")

        self.denoise_var = ctk.DoubleVar(value=self.settings.get("temporal_denoise", 0.4))
        self._slider_row(10, "Input Denoise", self.denoise_var, 0.0, 0.9, 18,
                         fmt=lambda v: "off" if v < 0.01 else f"{v:.2f}",
                         on_change=_live("temporal_denoise"),
                         desc="Smooths webcam grain before the model sees it, so it "
                              "stops reinventing detail over noise. Lower = steadier "
                              "but laggier input.")

        self.stillness_var = ctk.DoubleVar(value=self.settings.get("stillness_blend", 0.3))
        self._slider_row(11, "Stillness Blend", self.stillness_var, 0.0, 1.0, 20,
                         fmt=lambda v: "off" if v < 0.01 else f"{v:.2f}",
                         on_change=_live("stillness_blend"),
                         desc="While you hold still the engine re-feeds this much raw "
                              "webcam back into its own output. 0 disables it. Only "
                              "active when Freeze Filter is below 1.00.")

        self.clahe_var = ctk.BooleanVar(value=self.settings.get("normalize_lighting", False))
        self.clahe_cb = ctk.CTkSwitch(self.right_col, text="Normalize Lighting (CLAHE)",
                                      variable=self.clahe_var)
        self.clahe_cb.pack(anchor="w", padx=15, pady=(8, 4))

        self.cudagraph_var = ctk.BooleanVar(value=self.settings.get("cuda_graph", True))
        self.cudagraph_cb = ctk.CTkSwitch(self.right_col, text="Use CUDA Graphs",
                                          variable=self.cudagraph_var)
        self.cudagraph_cb.pack(anchor="w", padx=15, pady=4)

        # Expressions
        ctk.CTkLabel(self.advanced_col, text="Expression Overrides",
                     font=ctk.CTkFont(size=14, weight="bold")).pack(fill=ctk.X, padx=10, pady=(15, 5))
        
        self.expr_frame = ctk.CTkFrame(self.advanced_col, fg_color="transparent")
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
        ctk.CTkLabel(self.advanced_col, text="Trigger Sensitivities",
                     font=ctk.CTkFont(size=14, weight="bold")).pack(fill=ctk.X, padx=10, pady=(15, 5))
        
        self.sens_frame = ctk.CTkFrame(self.advanced_col, fg_color="transparent")
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

        # --- Session history ---
        hist_box = ctk.CTkFrame(self.system_col)
        hist_box.pack(fill=ctk.X, padx=10, pady=(10, 0))
        ctk.CTkLabel(hist_box, text="Session History", anchor="w",
                     font=ctk.CTkFont(size=13, weight="bold")).pack(
            fill=ctk.X, padx=8, pady=(6, 0))
        ctk.CTkLabel(hist_box,
                     text=f"Every START saves a snapshot of all settings "
                          f"(newest first, last {HISTORY_LIMIT} kept).",
                     font=ctk.CTkFont(size=11), text_color="gray",
                     justify="left", wraplength=340, anchor="w").pack(
            fill=ctk.X, padx=8, pady=(0, 4))

        self.history_var = ctk.StringVar(value="(no history yet)")
        self.history_menu = ctk.CTkOptionMenu(hist_box, variable=self.history_var,
                                              values=["(no history yet)"])
        self.history_menu.pack(fill=ctk.X, padx=8, pady=(0, 6))
        ctk.CTkButton(hist_box, text="↩ Restore These Settings",
                      command=self.on_history_restore).pack(
            fill=ctk.X, padx=8, pady=(0, 8))
        self._refresh_history_menu()

        ctk.CTkButton(self.system_col, text="🧪 Run Self-Test (verify live controls)",
                      command=self.run_selftest).pack(fill=ctk.X, padx=10, pady=(10, 0))

        # Logs
        ctk.CTkLabel(self.system_col, text="Engine Logs:", anchor="w").pack(
            fill=ctk.X, padx=10, pady=(10, 0))
        self.log_box = ctk.CTkTextbox(self.system_col, state="disabled", wrap="word",
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

        self.random_btn = ctk.CTkButton(self.right_container, text="🎲 Randomize Style (Ctrl+R)",
                                        command=self.randomize_prompt)
        self.random_btn.pack(fill=ctk.X, padx=10, pady=5)
        
        self.freeze_btn = ctk.CTkButton(self.right_container, text="❄️ Manual Freeze (F8)",
                                        command=self.toggle_freeze, fg_color=["#3a7ebf", "#1f538d"], hover_color=["#325882", "#14375e"])
        self.freeze_btn.pack(fill=ctk.X, padx=10, pady=5)
        
        self.snapshot_btn = ctk.CTkButton(self.right_container, text="🖼️ Take Snapshot (F12)",
                                          command=self.take_snapshot)
        self.snapshot_btn.pack(fill=ctk.X, padx=10, pady=5)
        
        self.replay_btn = ctk.CTkButton(self.right_container, text="📷 Save 5s WebP Replay (Ctrl+S)",
                                        command=self.save_replay)
        self.replay_btn.pack(fill=ctk.X, padx=10, pady=5)
        
        self.record_btn = ctk.CTkButton(self.right_container, text="🔴 Start Recording",
                                        command=self.toggle_recording, fg_color="#d9534f", hover_color="#c9302c")
        self.record_btn.pack(fill=ctk.X, padx=10, pady=5)

        self.start_btn = ctk.CTkButton(self.right_container, text="▶ START ENGINE",
                                       fg_color="#28a745", hover_color="#218838",
                                       command=self.start_script)
        self.start_btn.pack(fill=ctk.X, padx=10, pady=(20, 8))

        self.stop_btn = ctk.CTkButton(self.right_container, text="■ STOP", fg_color="#dc3545",
                                      hover_color="#c82333", state="disabled",
                                      command=self.stop_script)
        self.stop_btn.pack(fill=ctk.X, padx=10, pady=(0, 10))




        # --- VFX TAB ---
        self.vfx_scroll = ctk.CTkScrollableFrame(self.tab_vfx)
        self.vfx_scroll.pack(fill=ctk.BOTH, expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(self.vfx_scroll, text="Offline Post-Production VFX", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(0,10))
        ctk.CTkLabel(self.vfx_scroll, text="Fine-tune your VFX settings over a loaded .mp4 video, and then render the final perfectly composited video.", wraplength=350, justify="left").pack(pady=(0,20))
        
        self.vfx_vid_frame = ctk.CTkFrame(self.vfx_scroll, fg_color="transparent")
        self.vfx_vid_frame.pack(fill=ctk.X, pady=(0, 20))
        
        self.vfx_video_entry = ctk.CTkEntry(self.vfx_vid_frame, placeholder_text="Path to .mp4 video...")
        self.vfx_video_entry.pack(side=ctk.LEFT, fill=ctk.X, expand=True, padx=(0, 10))
        
        def _vfx_load_vid():
            from tkinter import filedialog
            path = filedialog.askopenfilename(title="Select Video", filetypes=[("Video Files", "*.mp4 *.mov *.avi *.mkv"), ("All Files", "*.*")])
            if path:
                self.vfx_video_entry.delete(0, ctk.END)
                self.vfx_video_entry.insert(0, path)
                # Automatically set it in the main camera entry so "START ENGINE" previews it!
                self.camera_entry.set(path)
                
        self.vfx_load_btn = ctk.CTkButton(self.vfx_vid_frame, text="📂 Load Video", width=100, command=_vfx_load_vid)
        self.vfx_load_btn.pack(side=ctk.RIGHT)
        
        def _send_vfx_op(v):
            if getattr(self, "cmd_socket", None):
                try: self.cmd_socket.send_string(__import__("json").dumps({"vfx_opacity": float(v)}))
                except Exception: pass
                
        self.vfx_op_var = ctk.DoubleVar(value=self.settings.get("vfx_opacity", 1.0))
        ctk.CTkLabel(self.vfx_scroll, text="VFX Opacity (Live Preview):", anchor="w").pack(anchor="w")
        vfx_slider = ctk.CTkSlider(self.vfx_scroll, variable=self.vfx_op_var, from_=0.0, to=1.0, number_of_steps=20, command=_send_vfx_op)
        vfx_slider.pack(fill=ctk.X, pady=(0,15))
        
        self.vfx_blend_var = ctk.StringVar(value=self.settings.get("vfx_blend_mode", "Normal"))
        def _send_vfx_blend(v):
            if getattr(self, "cmd_socket", None):
                try: self.cmd_socket.send_string(__import__("json").dumps({"vfx_blend_mode": v}))
                except Exception: pass
                
        ctk.CTkLabel(self.vfx_scroll, text="VFX Blend Mode:", anchor="w").pack(anchor="w")
        self.vfx_blend_menu = ctk.CTkOptionMenu(self.vfx_scroll, values=["Normal", "Screen", "Linear Dodge (Add)", "Color Dodge", "Overlay", "Soft Light", "Hard Light", "Multiply", "Darken", "Lighten", "Difference", "Exclusion"], variable=self.vfx_blend_var, command=_send_vfx_blend)
        self.vfx_blend_menu.pack(fill=ctk.X, pady=(0,15))
        
        self.no_face_track_var = ctk.BooleanVar(value=self.settings.get("no_face_track", False))
        self.no_face_track_cb = ctk.CTkSwitch(self.vfx_scroll, text="Full Frame Mode (Disable Face Tracking)", variable=self.no_face_track_var)
        self.no_face_track_cb.pack(anchor="w", pady=(0, 4))

        # Without this the selfie mask is applied twice — the input is
        # grey-screened before inference and the result is cut to your
        # silhouette during the HD paste-back — so the AI can never cover the
        # whole picture. Default off: existing behaviour is unchanged.
        self.no_segment_var = ctk.BooleanVar(value=self.settings.get("no_segment", False))
        self.no_segment_cb = ctk.CTkSwitch(
            self.vfx_scroll, text="Paint Whole Frame (Disable Cutout Mask)",
            variable=self.no_segment_var, command=self._send_no_segment)
        self.no_segment_cb.pack(anchor="w", pady=(0, 4))
        ctk.CTkLabel(self.vfx_scroll,
                     text="Pair with Full Frame Mode and Zoom 1.0 for a fully "
                          "painted picture. Disables background compositing.",
                     font=ctk.CTkFont(size=11), text_color="gray",
                     justify="left", wraplength=340).pack(anchor="w", pady=(0, 30))
        
        def _export_vfx():
            import subprocess
            video_path = self.vfx_video_entry.get().strip()
            if not video_path.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
                self.log("[Error] You must load a video file to export Offline VFX!")
                return
                
            self.save_settings()
            self.log(f"[Export] Starting Offline VFX Render for {video_path}...")
            try:
                subprocess.Popen([self.python_executable(), "process_video.py"])
                self.log("[Export] process_video.py launched in the background. Check console for progress.")
            except Exception as e:
                self.log(f"[Export] Error launching: {e}")
                
        self.export_btn = ctk.CTkButton(self.vfx_scroll, text="🎞️ Export Offline VFX",
                                        command=_export_vfx, height=40, font=ctk.CTkFont(size=14, weight="bold"),
                                        fg_color="#f0ad4e", hover_color="#ec971f")
        self.export_btn.pack(fill=ctk.X, pady=10)

        self.toggle_preview()
        self.update_video_frame()

    def _slider_row(self, row, label, var, lo, hi, steps, fmt, desc=None, on_change=None):
        r = row * 2
        ctk.CTkLabel(self.tune_frame, text=f"{label}:", anchor="w").grid(
            row=r, column=0, padx=2, pady=3, sticky="ew")
        value_label = ctk.CTkLabel(self.tune_frame, text=fmt(var.get()), width=90, anchor="e")
        value_label.grid(row=r, column=2, padx=2, pady=3, sticky="e")

        # Remember the row so a preset/history load can refresh the number.
        self._slider_rows.append((var, value_label, fmt))

        def _update(v):
            value_label.configure(text=fmt(v))
            if on_change:
                on_change(v)
                
        slider = ctk.CTkSlider(self.tune_frame, variable=var, from_=lo, to=hi,
                               number_of_steps=steps, width=110, command=_update)
        slider.grid(row=r, column=1, padx=2, pady=3, sticky="ew")
        self._slider_widgets[label] = slider
        
        if desc:
            desc_label = ctk.CTkLabel(self.tune_frame, text=desc, font=ctk.CTkFont(size=11), text_color="gray", justify="left", wraplength=350)
            desc_label.grid(row=r+1, column=0, columnspan=3, padx=2, pady=(0, 10), sticky="ew")

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
                    is_engine_frozen = False
                    if len(latest) > 0 and latest[0] in (0, 1):
                        is_engine_frozen = latest[0] == 1
                        latest = latest[1:]
                    
                    if is_engine_frozen:
                        self.freeze_btn.configure(text="[ AI FROZEN ]", fg_color="#d9534f", hover_color="#c9302c")
                    elif getattr(self, "manual_freeze", False):
                        self.freeze_btn.configure(text="⏸ Unfreeze (F8)", fg_color="#5cb85c", hover_color="#4cae4c")
                    else:
                        self.freeze_btn.configure(text="❄️ Manual Freeze (F8)", fg_color=["#3a7ebf", "#1f538d"], hover_color=["#325882", "#14375e"])

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
                        h_img, w_img = img_np.shape[:2]
                        if h_img > w_img:
                            new_h = side
                            new_w = int(w_img * (side / h_img))
                        else:
                            new_w = side
                            new_h = int(h_img * (side / w_img))
                        shown = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_AREA)
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
        if getattr(self, "easyneg_var", None) and self.easyneg_var.get():
            negative = negative + ", EasyNegative" if negative else "EasyNegative"
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
            
    def toggle_freeze(self):
        self.manual_freeze = not getattr(self, "manual_freeze", False)
        if getattr(self, "cmd_socket", None):
            import json
            self.cmd_socket.send_string(json.dumps({"manual_freeze": self.manual_freeze}))
            if self.manual_freeze:
                self.log("[Engine] Manual Freeze ON: Refining current frame.")
                self.freeze_btn.configure(text="▶ Unfreeze (F8)", fg_color="#5cb85c", hover_color="#4cae4c")
            else:
                self.log("[Engine] Manual Freeze OFF: Resumed webcam.")
                self.freeze_btn.configure(text="❄️ Manual Freeze (F8)", fg_color=["#3a7ebf", "#1f538d"], hover_color=["#325882", "#14375e"])
        else:
            self.log("[Engine] Cannot freeze: Engine is not running.")
            
    def take_snapshot(self):
        # We now send a command to the engine to save the high-res uncompressed frames!
        if getattr(self, "cmd_socket", None):
            import json
            self.cmd_socket.send_string(json.dumps({"save_snapshot": True}))
            self.log("[Engine] Requested high-res snapshot from engine...")
        else:
            self.log("[Engine] Cannot save snapshot: Engine is not running.")

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
                
                h, w = frames[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(filename, fourcc, 30.0, (w, h))
                
                for f in frames:
                    writer.write(f)
                writer.release()
                
                self.after(0, self.log, f"[Replay] Saved MP4 replay to {filename}")
            except Exception as e:
                self.after(0, self.log, f"[Replay] Error saving replay: {e}")
            finally:
                # Always release the flag, otherwise one failed save stops the
                # replay buffer refilling for the rest of the session.
                self.saving = False

        threading.Thread(target=_do_save, daemon=True).start()

    # -------------------------------------------------------------- logic --
    # (command payload, the log line the engine's handler emits) — the
    # self-test sends each one and waits for its echo, which proves the whole
    # round trip: socket, JSON, handler, and that the engine is still alive.
    def _selftest_probes(self):
        return [
            ({"composite": bool(self.bg_keep_var.get())}, "Composite Real Background:"),
            ({"no_segment": bool(self.no_segment_var.get())}, "Paint Whole Frame:"),
            ({"manual_freeze": False}, "Manual Freeze:"),
            ({"vfx_opacity": float(self.vfx_op_var.get())}, "VFX Opacity:"),
            ({"vfx_blend_mode": self.vfx_blend_var.get()}, "VFX Blend Mode:"),
            ({"zoom": float(self.zoom_var.get())}, "Zoom:"),
            ({"motion_smoothing": float(self.motion_var.get())}, "Motion Smoothing:"),
            ({"bokeh_blur": float(self.bokeh_var.get())}, "Bokeh Blur:"),
            ({"freeze_threshold": float(self.freeze_var.get())}, "Freeze Threshold:"),
            ({"guidance_scale": max(1.05, float(self.guidance_var.get()))}, "CFG:"),
            ({"sharpness": float(self.sharp_var.get())}, "Sharpness:"),
            ({"saturation": int(self.sat_var.get())}, "Saturation:"),
            ({"brightness": int(self.bright_var.get())}, "Brightness:"),
            ({"mask_feather": int(self.feather_var.get())}, "Mask Feather:"),
            ({"temporal_denoise": float(self.denoise_var.get())}, "Temporal Denoise:"),
            ({"stillness_blend": float(self.stillness_var.get())}, "Stillness Blend:"),
        ]

    def run_selftest(self):
        """Fire every live command with its CURRENT value and confirm the engine
        echoes each one back. Nothing changes; it only proves the wiring."""
        if self.process is None:
            self.log("[Self-test] Start the engine first.")
            return
        probes = self._selftest_probes()
        self._selftest_waiting = {marker: list(payload)[0] for payload, marker in probes}
        self._selftest_total = len(probes)
        self.log(f"[Self-test] Sending {len(probes)} live commands with their current "
                 f"values (nothing will change)...")
        unsent = []
        for payload, _marker in probes:
            if not self.send_command(payload):
                unsent.append(list(payload)[0])
        if unsent:
            self.log(f"[Self-test] Could not send: {', '.join(unsent)}")
        self.after(4000, self._selftest_report)

    def _selftest_report(self):
        missing = getattr(self, "_selftest_waiting", {})
        total = getattr(self, "_selftest_total", 0)
        passed = total - len(missing)
        if not missing:
            self.log(f"[Self-test] PASS — all {total} live controls answered.")
        else:
            self.log(f"[Self-test] {passed}/{total} answered. NO RESPONSE from: "
                     f"{', '.join(sorted(missing.values()))}")
            self.log("[Self-test] An unanswered command means the engine has no "
                     "handler for it — that control does nothing.")
        self._selftest_waiting = {}

    def _end_lora_swap(self):
        if not getattr(self, "_lora_swap_pending", False):
            return
        self._lora_swap_pending = False
        try:
            self.lora_dropdown.configure(state="normal")
        except Exception:
            pass

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

        waiting = getattr(self, "_selftest_waiting", None)
        if waiting:
            for marker in [m for m in waiting if m in message]:
                waiting.pop(marker, None)

        # The engine emits one of these when a hot-swap finishes either way.
        if getattr(self, "_lora_swap_pending", False) and (
                "hot-swap complete" in message or "Hot-swap to" in message):
            self._end_lora_swap()

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
        
        neg_val = self.neg_prompt_entry.get().strip()
        if getattr(self, "easyneg_var", None) and self.easyneg_var.get():
            neg_val = neg_val + ", EasyNegative" if neg_val else "EasyNegative"
            
        perf = self.perf_var.get()
        if perf.startswith("Maximum Quality"):
            fb, steps = 1, 4
        else:
            fb, steps = 1, 2
            
        cnet_val = self.controlnet_var.get()
        if cnet_val == "Depth (MiDaS)":
            cnet_cmd = "depth"
        elif cnet_val == "Canny Edge (Details)":
            cnet_cmd = "canny"
        elif cnet_val == "Lineart (Sketches)":
            cnet_cmd = "lineart"
        elif cnet_val == "OpenPose (Skeletal)":
            cnet_cmd = "openpose"
        elif cnet_val == "Depth + Canny (Heavy/Low FPS)":
            cnet_cmd = "multi"
        else:
            cnet_cmd = "none"

        cmd = [
            self.python_executable(), "-u",
            os.path.join(SCRIPT_DIR, "realtime_video.py"),
            "--prompt", self.prompt_entry.get(),
            "--negative_prompt", neg_val,
            "--camera", self.camera_entry.get().strip() or "0",
            "--lora", lora_val,
            "--controlnet", cnet_cmd,
            "--frame_buffer", str(fb),
            "--steps", str(steps),
            "--guidance_scale", f"{max(1.05, self.guidance_var.get()):.3f}",
            "--t_index", str(strength_to_t_index(self.strength_var.get())),
            "--freeze_threshold", f"{self.freeze_var.get():.3f}",
            "--motion_smoothing", f"{self.motion_var.get():.2f}",
            "--bokeh_blur", f"{self.bokeh_var.get():.2f}",
            "--zoom", f"{self.zoom_var.get():.2f}",
            "--expr_overrides", json.dumps({e: v.get() for e, v in self.expr_vars.items()}),
            "--sens_overrides", json.dumps({k: v.get() for k, v in self.sens_vars.items()}),
            "--zmq_port", str(zmq_port),
            "--cmd_port", str(cmd_port)
        ]
        if self.mirror_var.get():
            cmd.append("--mirror_camera")
        if self.vcam_var.get():
            cmd.append("--virtual_camera")
        cmd += [
            "--sharpness", f"{self.sharp_var.get():.3f}",
            "--saturation", str(int(self.sat_var.get())),
            "--brightness", str(int(self.bright_var.get())),
            "--mask_feather", str(int(self.feather_var.get())),
            "--temporal_denoise", f"{self.denoise_var.get():.3f}",
            "--stillness_blend", f"{self.stillness_var.get():.3f}",
        ]
        if self.no_segment_var.get():
            cmd.append("--no_segment")
        if self.no_face_track_var.get():
            cmd.append("--no_face_track")
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
        self.no_face_track_cb.configure(state=state)
        self.audio_cb.configure(state=state)
        self.bg_keep_cb.configure(state=state)
        self.bg_entry.configure(state=state)
        self.bg_btn.configure(state=state)
        self.bg_clear_btn.configure(state=state)
        self.clahe_cb.configure(state=state)
        self.cudagraph_cb.configure(state=state)
        # Read once at launch by the engine, so they must not look live.
        self.perf_dropdown.configure(state=state)
        self.controlnet_dropdown.configure(state=state)
        for name in ("AI Strength",):
            w = self._slider_widgets.get(name)
            if w is not None:
                w.configure(state=state)

    def start_script(self):
        if self.process is not None and self.process.poll() is None:
            return

        self.save_settings()
        self.append_history()
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
