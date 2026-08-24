import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

# Mock sys.argv so build_args() doesn't crash on missing flags
sys.argv = ["compile_worker.py", "--steps", "4", "--t_index", "14"]

from realtime_video import build_args, load_model_and_engine

def compile_process():
    args = build_args()
    pipe, stream = load_model_and_engine(args, args.lora)

if __name__ == "__main__":
    compile_process()
