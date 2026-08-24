import sys
import threading
import time

def close_app(app):
    time.sleep(2)
    print("UI loaded successfully! Closing...")
    app.quit()

try:
    from launcher import VTuberStudioApp
    app = VTuberStudioApp()
    threading.Thread(target=close_app, args=(app,), daemon=True).start()
    app.mainloop()
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)