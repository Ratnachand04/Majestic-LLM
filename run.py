"""Start Majestic and keep it running.

    python run.py

Opens http://localhost:8000/ and stays up until Ctrl-C. Everything runs
offline: no GPU, no downloads, no API keys.
"""
from __future__ import annotations

import argparse
import sys
import threading
import webbrowser


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Majestic LLM studio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="auto-reload on edits")
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser")
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        print("error: uvicorn is not installed.", file=sys.stderr)
        print("  uv pip install fastapi uvicorn   (or: pip install fastapi uvicorn)",
              file=sys.stderr)
        return 2

    url = f"http://{'localhost' if args.host in ('127.0.0.1', '0.0.0.0') else args.host}:{args.port}/"

    print()
    print("  Majestic LLM")
    print(f"  studio   {url}")
    print(f"  api docs {url}docs")
    print("  Ctrl-C to stop")
    print()

    if not args.no_browser:
        # Fire after the server is accepting connections, not before, or the
        # browser races the bind and shows a connection error.
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    uvicorn.run(
        "api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
