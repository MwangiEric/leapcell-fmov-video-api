#!/bin/sh
set -e

# Install FFmpeg (lightweight video engine)
apt-get update && apt-get install -y --no-install-recommends ffmpeg

# Install Python dependencies (fast, no heavy compilation)
pip install fastapi uvicorn fmov pillow python-multipart pydantic aiohttp

# Verify FFmpeg installed
ffmpeg -version | head -1
