import audio_sync, time
tracker = audio_sync.AudioTracker()
time.sleep(2)
tracker.close()
print("Audio successful.")
