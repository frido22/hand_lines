"""
Hand Lines — API

Direct HTTP calls to OpenRouter. No SDK needed.
"""

import os
import base64
import re

import cv2
import numpy as np
import requests


API_BASE = "https://openrouter.ai/api/v1/chat/completions"


def _get_key():
    return os.environ.get("OPENROUTER_API_KEY", "")


def _encode_image(image_rgb, quality=85):
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode('utf-8')


def _call(model, messages, max_tokens=1024, temperature=0.8):
    key = _get_key()
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set")

    resp = requests.post(API_BASE, headers={
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }, json={
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }, timeout=90)

    if resp.status_code != 200:
        raise RuntimeError(f"API error {resp.status_code}: {resp.text[:300]}")

    return resp.json().get("choices", [{}])[0].get("message", {})


# --- Gemini palm line image ---

HIGHLIGHT_PROMPT = """Edit this palm photo. Overlay thin black lines on the actual visible creases of the palm. Draw exactly these 4 lines:

1. HEART LINE — runs horizontally across the TOP of the palm, just below the fingers, from the pinky side to the index finger side. Label it "HEART LINE".

2. HEAD LINE — runs horizontally across the MIDDLE of the palm, below the heart line, starting near the thumb/index gap and going across. Label it "HEAD LINE".

3. LIFE LINE — curves from between the thumb and index finger DOWNWARD in an arc around the base of the thumb. Label it "LIFE LINE".

4. FATE LINE — runs VERTICALLY up the center of the palm from the wrist toward the middle finger. Label it "FATE LINE".

Rules:
- All lines must be thin, clean BLACK lines
- Trace the lines ON the actual visible creases in the photo
- Add small black text labels with arrows pointing to each line
- Keep the original palm photo fully visible underneath
- No other decorations, symbols, colors, or region labels"""


def generate_palm_image(image_rgb):
    """Send palm to Gemini to draw lines on it. Returns RGB numpy array."""
    b64 = _encode_image(image_rgb, quality=90)

    msg = _call(
        model="google/gemini-2.5-flash-image",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": HIGHLIGHT_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }],
        max_tokens=4096,
    )

    return _extract_image(msg)


def _extract_image(msg):
    """Extract image from OpenRouter response — checks 'images' field and 'content'."""
    # OpenRouter returns images in a separate 'images' array
    images = msg.get("images", [])
    for part in images:
        if isinstance(part, dict) and part.get("type") == "image_url":
            url = part.get("image_url", {}).get("url", "")
            img = _decode_data_url(url)
            if img is not None:
                return img

    # Fallback: check content field
    content = msg.get("content")
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                url = part.get("image_url", {}).get("url", "")
                img = _decode_data_url(url)
                if img is not None:
                    return img

    if isinstance(content, str):
        match = re.search(r'data:image/[^;]+;base64,([A-Za-z0-9+/=]+)', content)
        if match:
            return _decode_b64(match.group(1))

    raise RuntimeError("No image in Gemini response")


def _decode_data_url(url):
    if not url:
        return None
    match = re.search(r'base64,(.+)', url)
    return _decode_b64(match.group(1)) if match else None


def _decode_b64(b64_string):
    img_bytes = base64.b64decode(b64_string)
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("Failed to decode image")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# --- Fortune text ---

FORTUNE_SYSTEM = """You are a wise palm reader. Examine the palm photos carefully and give a reading for each visible line.

Return ONLY this markdown format — no introduction, no closing, just the lines:

**Heart Line** — [2-3 sentences. Describe what you see (deep, faint, curved, straight, long, short, branching?) and what it reveals about love and emotional nature.]

**Head Line** — [2-3 sentences. Describe the line and what it means for intellect, career, decision-making.]

**Life Line** — [2-3 sentences. Describe the curve and depth. What does it say about vitality and energy?]

**Fate Line** — [2-3 sentences. If faint or absent, interpret that meaningfully.]

**Sun Line** — [1-2 sentences about creativity and success, only if visible.]

**Mercury Line** — [1-2 sentences about communication, only if visible.]

Rules:
- Be SPECIFIC about what you observe (line depth, curvature, length, branches)
- When both hands shown: left = inner potential, right = lived experience. Note contrasts.
- Warm, poetic tone — not generic
- Skip Sun/Mercury if not visible
- Do NOT add any opening or closing sentences, just the line readings"""


def generate_fortune(left_img=None, right_img=None):
    content = [{"type": "text", "text": "Read these palms and give a fortune."}]

    for label, img in [("Left hand:", left_img), ("Right hand:", right_img)]:
        if img is not None and isinstance(img, np.ndarray) and img.size > 0:
            b64 = _encode_image(img)
            content.append({"type": "text", "text": label})
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

    msg = _call(
        model="moonshotai/kimi-k2.5",
        messages=[
            {"role": "system", "content": FORTUNE_SYSTEM},
            {"role": "user", "content": content},
        ],
        max_tokens=4096,
        temperature=0.85,
    )

    text = msg.get("content", "")
    if isinstance(text, str) and len(text.strip()) > 20:
        return text.strip()

    raise RuntimeError("Fortune LLM returned empty response")
