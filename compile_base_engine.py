import os
import sys
import subprocess

def main():
    print("=====================================================")
    print("Starting Overnight TensorRT Base Engine Compilation...")
    print("=====================================================")
    print("This will build the massive 4-Step Batch-8 Engine.")
    print("Failsafe activated: If the Nvidia driver deadlocks, it will auto-kill after 3 hours.")
    print("When this finishes (or fails), the computer will automatically sleep.")
    print("=====================================================")
    
    # We use subprocess to isolate the TRT compiler and avoid Windows multiprocessing bugs
    script_path = os.path.join(os.path.dirname(__file__), "compile_worker.py")
    
    try:
        # Wait up to 3 hours (10,800 seconds)
        subprocess.run([sys.executable, script_path], timeout=10800, check=True)
        print("\n=====================================================")
        print("COMPILATION FINISHED SUCCESSFULLY!")
        print("The timing cache and engine are permanently saved.")
        print("=====================================================")
    except subprocess.TimeoutExpired:
        print("\n[!] TIMEOUT REACHED: The Nvidia WDDM driver deadlocked!")
        print("[!] Terminated the frozen compiler to save your GPU...")
        print("[!] The engine failed to build. You must use 2-Step Mode tomorrow.")
    except subprocess.CalledProcessError as e:
        print(f"\n[!] The compiler crashed with an error code: {e.returncode}")
        print("[!] Please check the console output above to see what went wrong.")
    
    print("\nSleeping the PC...")
    os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")

if __name__ == "__main__":
    main()
