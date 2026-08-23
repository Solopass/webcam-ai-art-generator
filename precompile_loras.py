import os
import subprocess
import time
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
loras_dir = os.path.join(SCRIPT_DIR, "loras")

print("=======================================")
print("  LoRA TensorRT Batch Pre-Compiler")
print("=======================================\n")

if not os.path.exists(loras_dir):
    os.makedirs(loras_dir)

lora_files = sorted([f for f in os.listdir(loras_dir) if f.endswith(".safetensors")])

if not lora_files:
    print("No .safetensors files found in the 'loras/' directory.")
    sys.exit(0)

print(f"Found {len(lora_files)} LoRAs in directory.")

for lora in lora_files:
    safe = "".join(c for c in lora if c.isalnum() or c in ("-", "_")).replace("safetensors", "")
    engine_dir = os.path.join(SCRIPT_DIR, f"engines_tinyvae_{safe}_fb1")
    
    if os.path.exists(os.path.join(engine_dir, "unet.engine")):
        print(f"[\u2713] {lora} is already compiled. Skipping.")
        continue
        
    print(f"\n======================================")
    print(f"Compiling {lora}")
    print(f"This will take 5-15 minutes. DO NOT CLOSE.")
    print(f"======================================")
    
    cmd = [
        os.path.join(SCRIPT_DIR, "venv", "Scripts", "python.exe"), "-u", "realtime_video.py",
        "--mock_camera",
        "--lora", lora,
        "--cmd_port", "12345"
    ]
    
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    
    import zmq
    ctx = zmq.Context()
    cmd_sock = ctx.socket(zmq.PUSH)
    cmd_sock.connect("tcp://127.0.0.1:12345")
    
    ready = False
    for line in iter(proc.stdout.readline, ""):
        print(f"  {line}", end="")
        if "[Engine] READY" in line:
            print(f"\n[\u2713] Compilation successful for {lora}! Gracefully shutting down...")
            time.sleep(1)
            try:
                cmd_sock.send_json({"cmd": "stop"})
            except Exception:
                pass
            ready = True
            
    proc.wait()
    cmd_sock.close()
    
    if ready:
        print(f"[\u2713] Finished compiling {lora}.")
    else:
        print(f"[!] {lora} FAILED to compile!")
        
print("\n=======================================")
print("All LoRAs are pre-compiled and ready for streaming!")
print("=======================================")