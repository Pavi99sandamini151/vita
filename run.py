import subprocess
import sys
from pathlib import Path

def run_servers():
    project_root = Path(__file__).parent
    frontend_path = project_root / "frontend" / "main.py"
    backend_path = project_root / "backend" / "app.py"

    try:
        # Start frontend
        frontend_process = subprocess.Popen([sys.executable, str(frontend_path)])
        
        # Start backend
        backend_process = subprocess.Popen([sys.executable, str(backend_path)])
        
        # Wait for both processes
        frontend_process.wait()
        backend_process.wait()
        
    except KeyboardInterrupt:
        print("\nShutting down servers...")
        frontend_process.terminate()
        backend_process.terminate()

if __name__ == "__main__":
    run_servers()