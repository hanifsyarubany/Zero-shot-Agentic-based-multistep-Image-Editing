import os
import io
import base64
import asyncio
import textwrap
from typing import List, Optional, Tuple

import requests
from PIL import Image, ImageDraw, ImageFont

import chainlit as cl

"""
Chainlit minimal frontend for image editing.
- Users upload images via the chat paperclip (no extra prompts).
- Bot replies with **images only** (exactly one PNG per turn).
- First turn must include 1–2 images; otherwise we reply with an image card telling them to upload.
- Subsequent turns can be text-only (continue editing the latest context via /sent-only-text-query).
"""

# -----------------------------
# Config
# -----------------------------
BASE_URL = os.getenv("IMAGE_EDIT_API_BASE_URL", "http://127.0.0.1:8000")
TWO_IMG_EP = f"{BASE_URL}/sent-two-images"
ONE_IMG_EP = f"{BASE_URL}/sent-one-image"
TEXT_ONLY_EP = f"{BASE_URL}/sent-only-text-query"

# Reasonable timeouts for heavy diffusion jobs
TIMEOUT_TWO = float(os.getenv("TIMEOUT_TWO", 600))
TIMEOUT_ONE = float(os.getenv("TIMEOUT_ONE", 600))
TIMEOUT_TEXT = float(os.getenv("TIMEOUT_TEXT", 600))

# -----------------------------
# Helpers
# -----------------------------

def sanitize_base64(data: str) -> str:
    """Strip a data URL header if present and return raw base64 string."""
    if not data:
        return data
    if data.startswith("data:"):
        try:
            return data.split(",", 1)[1]
        except Exception:
            pass
    return data


def file_to_base64(file: cl.File) -> str:
    """Accept Chainlit File/Image-like element and return a raw base64 string."""
    # Prefer in-memory bytes when available
    data: bytes
    if getattr(file, "content", None):
        data = file.content
    else:
        with open(file.path, "rb") as f:
            data = f.read()
    return base64.b64encode(data).decode("utf-8")


def b64_to_png_bytes(b64: str) -> Tuple[bytes, str]:
    """Decode unknown base64 image → PNG bytes. Returns (bytes, mime)."""
    raw = base64.b64decode(sanitize_base64(b64))
    with Image.open(io.BytesIO(raw)) as im:
        if im.mode not in ("RGB", "RGBA"):
            im = im.convert("RGB")
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue(), "image/png"


def info_image_bytes(msg: str, w: int = 960, h: int = 360) -> Tuple[bytes, str]:
    """Create a plain PNG containing the message (so replies are always images)."""
    img = Image.new("RGB", (w, h), color=(245, 246, 250))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 28)
    except Exception:
        font = ImageFont.load_default()

    wrapped: List[str] = []
    for line in msg.splitlines():
        segs = textwrap.wrap(line, width=48) or [""]
        wrapped.extend(segs)
    text = "\n".join(wrapped)

    bbox = draw.multiline_textbbox((0, 0), text, font=font, align="center")
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = (w - tw) // 2
    y = (h - th) // 2
    draw.multiline_text((x, y), text, fill=(40, 40, 40), font=font, align="center")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue(), "image/png"


async def _post_json(url: str, payload: dict, timeout: float) -> requests.Response:
    """Run blocking requests.post in a worker thread to keep UI responsive."""
    def _do():
        return requests.post(url, json=payload, timeout=timeout)
    return await asyncio.to_thread(_do)


async def call_two_images_and_text(user_query: str, img1_b64: str, img2_b64: str) -> str:
    payload = {
        "req": {"user_query": user_query},
        "img1": {"base64_img": sanitize_base64(img1_b64)},
        "img2": {"base64_img": sanitize_base64(img2_b64)},
    }
    r = await _post_json(TWO_IMG_EP, payload, TIMEOUT_TWO)
    if r.status_code != 200:
        raise RuntimeError(f"API {TWO_IMG_EP} error {r.status_code}: {r.text}")
    return r.json()["response"]


async def call_one_image_and_text(user_query: str, img_b64: str) -> str:
    payload = {
        "req": {"user_query": user_query},
        "img": {"base64_img": sanitize_base64(img_b64)},
    }
    r = await _post_json(ONE_IMG_EP, payload, TIMEOUT_ONE)
    if r.status_code != 200:
        raise RuntimeError(f"API {ONE_IMG_EP} error {r.status_code}: {r.text}")
    return r.json()["response"]


async def call_text_only(user_query: str) -> str:
    payload = {"user_query": user_query}
    r = await _post_json(TEXT_ONLY_EP, payload, TIMEOUT_TEXT)
    if r.status_code != 200:
        raise RuntimeError(f"API {TEXT_ONLY_EP} error {r.status_code}: {r.text}")
    return r.json()["response"]


async def process_turn(user_query: str, files: Optional[List[cl.File]]) -> Tuple[bytes, str]:
    """Route based on number of images; always return a PNG bytes result."""
    primed = bool(cl.user_session.get("primed") or False)
    files = files or []

    # Keep at most 2 images
    images: List[cl.File] = []
    for el in files:
        mime = getattr(el, "mime", "") or getattr(el, "type", "")
        is_image = isinstance(mime, str) and mime.startswith("image/")
        if is_image or hasattr(el, "content") or hasattr(el, "path"):
            images.append(el)
        if len(images) == 2:
            break

    n = len(images)

    try:
        if n >= 2:
            img1_b64 = file_to_base64(images[0])
            img2_b64 = file_to_base64(images[1])
            result_b64 = await call_two_images_and_text(user_query, img1_b64, img2_b64)
            cl.user_session.set("primed", True)
        elif n == 1:
            img_b64 = file_to_base64(images[0])
            result_b64 = await call_one_image_and_text(user_query, img_b64)
            cl.user_session.set("primed", True)
        else:
            if not primed:
                return info_image_bytes("Upload 1–2 images with your message to start editing.")
            result_b64 = await call_text_only(user_query)

        return b64_to_png_bytes(result_b64)
    except Exception as e:
        return info_image_bytes(f"Backend error:\n{type(e).__name__}: {e}")


# -----------------------------
# Chainlit lifecycle
# -----------------------------

@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("primed", False)


@cl.on_chat_resume
async def on_chat_resume():
    if cl.user_session.get("primed") is None:
        cl.user_session.set("primed", False)


@cl.on_message
async def on_message(message: cl.Message):
    # Collect any files attached via the chat paperclip icon
    files: List[cl.File] = []
    for el in (message.elements or []):
        mime = getattr(el, "mime", "") or getattr(el, "type", "")
        is_image = isinstance(mime, str) and mime.startswith("image/")
        if is_image or hasattr(el, "content") or hasattr(el, "path"):
            files.append(el)

    text = (message.content or "").strip()

    # Process and send exactly one image back
    png_bytes, mime = await process_turn(text, files)
    await cl.Message(
        content="",  # image-only response
        elements=[cl.Image(name="result.png", content=png_bytes, mime=mime, display="inline")],
    ).send()
