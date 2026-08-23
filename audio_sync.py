import sounddevice as sd
import numpy as np
import time

class AudioTracker:
    def __init__(self):
        self.volume = 0.0
        self.is_speaking = False
        self.last_spoken_time = 0.0
        self.threshold = 1.5
        try:
            self.stream = sd.InputStream(callback=self.audio_callback)
            self.stream.start()
        except Exception as e:
            print("Audio error:", e)
            self.stream = None
            
    def set_threshold(self, threshold):
        self.threshold = threshold
        
    def audio_callback(self, indata, frames, time_info, status):
        volume_norm = np.linalg.norm(indata) * 10
        self.volume = volume_norm
        
        if self.volume > self.threshold:
            self.last_spoken_time = time.time()
            self.is_speaking = True
        elif time.time() - self.last_spoken_time > 0.5:
            # Only toggle to False if there has been silence for 0.5 seconds (debouncing)
            self.is_speaking = False

    def close(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
