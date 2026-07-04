"""Vercel Python entry point: re-exports the FastAPI app defined in backend/main.py."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.main import app  # noqa: E402
