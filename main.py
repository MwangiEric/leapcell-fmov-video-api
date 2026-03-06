import asyncio
import base64
import io
import os
import tempfile
from typing import List, Optional, Union, Literal

import aiohttp
from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fmov import Video
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field

app = FastAPI(title="Video Editing API (fmov)", version="1.0.0")

# Ensure tmp exists
os.makedirs("./tmp", exist_ok=True)


# ============ JSON SCHEMAS (Editly-Style) ============

class Position(BaseModel):
    x: Optional[int] = Field(None, description="Center if null")
    y: Optional[int] = Field(None, description="Center if null")


class TextLayer(BaseModel):
    type: Literal["text"] = "text"
    text: str
    font_size: int = 70
    color: str = "white"
    position: Position = Position()
    start_time: float = 0.0
    duration: float = 10.0
    background_color: Optional[str] = None


class ImageLayer(BaseModel):
    type: Literal["image"] = "image"
    url: str  # URL, base64, or filename reference
    position: Position = Position()
    width: Optional[int] = None
    height: Optional[int] = None
    start_time: float = 0.0
    duration: float = 10.0


class AudioTrack(BaseModel):
    url: str
    start_time: float = 0.0
    volume: float = 1.0


class VideoClip(BaseModel):
    source: str  # "upload", URL, or base64
    start_time: float = 0.0
    duration: Optional[float] = None  # None = auto-detect


class EditSpec(BaseModel):
    """Main JSON spec - just like Editly"""
    width: int = 1920
    height: int = 1080
    fps: int = 30
    clips: List[VideoClip] = []
    layers: List[Union[TextLayer, ImageLayer]] = []
    audio_tracks: List[AudioTrack] = []
    background_color: str = "#000000"


# ============ VIDEO ENGINE ============

class VideoEngine:
    def __init__(self):
        self.tmp = "./tmp"
        # Try to find a system font
        self.font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/System/Library/Fonts/Helvetica.ttc",  # macOS fallback
        ]
        self.font_path = self._find_font()

    def _find_font(self):
        for path in self.font_paths:
            if os.path.exists(path):
                return path
        return None

    async def render(self, spec: EditSpec, uploaded_files: dict = None) -> str:
        """Render video from JSON spec"""
        output = os.path.join(self.tmp, f"video_{id(spec)}.mp4")
        
        # Calculate duration
        duration = 10.0
        if spec.clips and spec.clips[0].duration:
            duration = spec.clips[0].duration
        
        total_frames = int(duration * spec.fps)

        with Video((spec.width, spec.height), framerate=spec.fps, path=output) as video:
            # Render frames
            for frame_idx in range(total_frames):
                current_time = frame_idx / spec.fps
                
                # Create base frame
                img = Image.new("RGB", (spec.width, spec.height), spec.background_color)
                
                # Render active layers
                for layer in spec.layers:
                    if self._is_active(layer, current_time):
                        img = await self._render_layer(img, layer, uploaded_files)
                
                video.pipe(img)

            # Add audio tracks
            for audio in spec.audio_tracks:
                audio_path = await self._get_file(audio.url, uploaded_files, f"audio_{id(audio)}")
                if audio_path:
                    video.sound_at_second(audio.start_time, audio_path, audio.volume)

        return output

    def _is_active(self, layer, t: float) -> bool:
        start = getattr(layer, 'start_time', 0)
        dur = getattr(layer, 'duration', 10)
        return start <= t < (start + dur)

    async def _render_layer(self, img, layer, files):
        if isinstance(layer, TextLayer):
            return self._render_text(img, layer)
        elif isinstance(layer, ImageLayer):
            return await self._render_image(img, layer, files)
        return img

    def _render_text(self, img: Image.Image, layer: TextLayer):
        draw = ImageDraw.Draw(img)
        
        # Load font
        try:
            if self.font_path:
                font = ImageFont.truetype(self.font_path, layer.font_size)
            else:
                font = ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()

        # Position
        x = layer.position.x if layer.position.x is not None else img.width // 2
        y = layer.position.y if layer.position.y is not None else img.height // 2

        # Background
        if layer.background_color:
            bbox = draw.textbbox((0, 0), layer.text, font=font)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            pad = 10
            draw.rectangle([x-w//2-pad, y-h//2-pad, x+w//2+pad, y+h//2+pad], 
                          fill=layer.background_color)

        # Shadow + text
        draw.text((x+2, y+2), layer.text, fill="black", font=font, anchor="mm")
        draw.text((x, y), layer.text, fill=layer.color, font=font, anchor="mm")
        
        return img

    async def _render_image(self, img, layer, files):
        # Get image data
        overlay = None
        
        if layer.url.startswith("data:"):
            # Base64
            data = base64.b64decode(layer.url.split(",", 1)[1])
            overlay = Image.open(io.BytesIO(data)).convert("RGBA")
        elif files and layer.url in files:
            # Uploaded file
            overlay = Image.open(io.BytesIO(files[layer.url])).convert("RGBA")
        else:
            # URL - download
            path = await self._download(layer.url, f"img_{id(layer)}.png")
            overlay = Image.open(path).convert("RGBA")

        # Resize
        if layer.width and layer.height:
            overlay = overlay.resize((layer.width, layer.height))

        # Position
        x = layer.position.x if layer.position.x is not None else (img.width - overlay.width) // 2
        y = layer.position.y if layer.position.y is not None else (img.height - overlay.height) // 2

        # Composite
        img.paste(overlay, (x, y), overlay)
        return img

    async def _get_file(self, url, files, prefix):
        """Get file path from URL, files dict, or download"""
        if url.startswith("data:"):
            path = os.path.join(self.tmp, f"{prefix}.mp3")
            data = base64.b64decode(url.split(",", 1)[1])
            with open(path, 'wb') as f:
                f.write(data)
            return path
        elif files and url in files:
            path = os.path.join(self.tmp, f"{prefix}.mp3")
            with open(path, 'wb') as f:
                f.write(files[url])
            return path
        else:
            return await self._download(url, f"{prefix}.mp3")

    async def _download(self, url: str, filename: str) -> str:
        path = os.path.join(self.tmp, filename)
        if os.path.exists(path):
            return path
        
        async with aiohttp.ClientSession() as s:
            async with s.get(url) as r:
                with open(path, 'wb') as f:
                    f.write(await r.read())
        return path


engine = VideoEngine()


# ============ API ENDPOINTS ============

@app.get("/", response_class=HTMLResponse)
async def index():
    """Simple HTML form for testing"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Video Editing API</title>
        <style>
            body { font-family: sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
            textarea { width: 100%; height: 300px; font-family: monospace; }
            button { padding: 10px 20px; background: #007bff; color: white; border: none; cursor: pointer; }
        </style>
    </head>
    <body>
        <h1>🎬 Video Editing API (fmov)</h1>
        <p>Editly-style JSON interface. <a href="/template">Get template</a></p>
        
        <form action="/render" method="post" enctype="multipart/form-data">
            <h3>JSON Spec:</h3>
            <textarea name="spec_json">{
  "width": 1280,
  "height": 720,
  "fps": 30,
  "clips": [{"source": "upload", "duration": 5}],
  "layers": [
    {"type": "text", "text": "Hello World", "font_size": 80, "color": "yellow"}
  ]
}</textarea>
            
            <h3>Upload Video (optional):</h3>
            <input type="file" name="file" accept="video/*"><br><br>
            
            <button type="submit">Render Video</button>
        </form>
        
        <h2>Or use JSON directly:</h2>
        <pre>curl -X POST https://your-app.leapcell.dev/render \
  -H "Content-Type: application/json" \
  -d '{"width":1280,"height":720,"layers":[{"type":"text","text":"Hi"}]}' \
  --output video.mp4</pre>
    </body>
    </html>
    """


@app.get("/template")
async def get_template():
    """Return example JSON template"""
    return {
        "description": "Editly-style JSON spec",
        "template": {
            "width": 1920,
            "height": 1080,
            "fps": 30,
            "clips": [
                {"source": "https://example.com/video.mp4", "duration": 10}
            ],
            "layers": [
                {
                    "type": "text",
                    "text": "Hello World",
                    "font_size": 80,
                    "color": "yellow",
                    "position": {"x": 960, "y": 540},
                    "start_time": 0,
                    "duration": 5
                },
                {
                    "type": "image",
                    "url": "https://example.com/logo.png",
                    "position": {"x": 100, "y": 100},
                    "width": 200,
                    "height": 100,
                    "start_time": 2,
                    "duration": 3
                }
            ],
            "audio_tracks": [
                {"url": "https://example.com/music.mp3", "start_time": 0, "volume": 0.5}
            ]
        }
    }


@app.post("/render")
async def render_json(spec: EditSpec):
    """
    Render from JSON (URLs only, no file uploads)
    """
    try:
        output = await engine.render(spec)
        return FileResponse(output, media_type="video/mp4", filename="rendered.mp4")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/render/upload")
async def render_upload(
    spec_json: str = Form(...),
    file: Optional[UploadFile] = File(None)
):
    """
    Render with file upload
    """
    try:
        spec = EditSpec.parse_raw(spec_json)
        
        files = {}
        if file:
            content = await file.read()
            files["upload"] = content  # Reference as "upload" in JSON
        
        output = await engine.render(spec, files)
        return FileResponse(output, media_type="video/mp4", filename="rendered.mp4")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "engine": "fmov",
        "memory": "512mb-optimized",
        "features": ["text-overlay", "image-overlay", "audio-mix", "json-api"]
    }
