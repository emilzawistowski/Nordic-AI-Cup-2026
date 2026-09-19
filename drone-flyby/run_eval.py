import subprocess
import time
import requests
import sys

proc = subprocess.Popen(["./.venv/bin/python", "api.py"])
try:
    for i in range(30):
        try:
            r = requests.get("http://localhost:9053/", timeout=1)
            if r.status_code < 500:
                print("Server is up!")
                break
        except Exception:
            pass
        time.sleep(1)
    
    cmd = ["./.venv/bin/python", "local_evaluator.py"]
    if "--realtime" in sys.argv:
        cmd.append("--realtime")
    
    res = subprocess.run(cmd)
finally:
    proc.terminate()
    proc.wait()
