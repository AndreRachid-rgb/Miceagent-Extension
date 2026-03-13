"""Entry point for miceagent backend. Run with: uv run server.py"""
import sys
import os

# Ensure the backend directory is in sys.path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn

if __name__ == "__main__":
    uvicorn.run("src.main:app", host="127.0.0.1", port=8000, reload=True)
