"""Entry point for miceagent backend. Run with: uv run server.py"""
import sys
import os
import argparse

# Ensure the backend directory is in sys.path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()
    uvicorn.run("src.main:app", host=args.host, port=args.port, reload=True)
