"""
Hand Lines — Gradio App
"""

import gradio as gr
import numpy as np

from cv_pipeline import detect_hand, landmarks_to_pixels, get_palm_crop_box, create_palm_mask
from api import generate_palm_image, generate_fortune


def crop_palm(image):
    if image is None:
        return None
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)

    landmarks = detect_hand(image)
    if landmarks is None:
        return None

    h, w = image.shape[:2]
    pts = landmarks_to_pixels(landmarks, w, h)
    x1, y1, x2, y2 = get_palm_crop_box(pts, w, h)

    mask = create_palm_mask(pts, (h, w))
    mask_f = mask[:, :, np.newaxis].astype(np.float32) / 255.0
    isolated = (image.astype(np.float32) * mask_f).astype(np.uint8)

    return isolated[y1:y2, x1:x2]


def on_capture(frame, step, left_img, right_img):
    if frame is None or step >= 2:
        return left_img, right_img, step, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

    if len(frame.shape) == 2:
        frame = np.stack([frame] * 3, axis=-1)

    cropped = crop_palm(frame)
    if cropped is None:
        return (
            left_img, right_img, step,
            gr.update(), gr.update(), gr.update(), gr.update(),
            "No hand detected — hold your palm flat and open",
            None,
        )

    if step == 0:
        return (
            cropped, right_img, 1,
            "Now capture your **right palm**",
            cropped, gr.update(),
            gr.update(interactive=False), "",
            None,
        )
    else:
        return (
            left_img, cropped, 2,
            "Both palms captured — press **Read My Palm**",
            gr.update(), cropped,
            gr.update(interactive=True), "",
            None,
        )


def on_read_start():
    return (
        "Reading your palms...",
        gr.update(interactive=False, value="Reading..."),
    )


def on_read(left_img, right_img):
    try:
        left_vis = generate_palm_image(left_img) if left_img is not None else None
        right_vis = generate_palm_image(right_img) if right_img is not None else None
        fortune = generate_fortune(left_img, right_img)
        return left_vis, right_vis, fortune, "", gr.update(value="Read My Palm", interactive=True)
    except Exception as e:
        return gr.update(), gr.update(), "", str(e), gr.update(value="Read My Palm", interactive=True)


def on_reset():
    return (
        None, None, 0,
        "Capture your **left palm**",
        None, None,
        gr.update(interactive=False),
        "", "", None,
    )


BG = "#2e2419"
BG_CARD = "#362c20"
ACCENT = "#c87a5a"
TEXT = "#e0d0bc"
TEXT_MID = "#a89880"
TEXT_LIGHT = "#8a7a68"
LAVENDER = "#3a3028"
SAND = "#4a3d30"

css = f"""
/* Seamless warm background */
html, body, gradio-app,
body > div, .main, .wrap, .app, #root, .contain,
.gradio-container, .dark {{
    background: {BG} !important;
    margin: 0; padding: 0;
}}

.gradio-container {{
    max-width: 680px !important;
    margin: 0 auto !important;
    padding: 12px 16px !important;
}}

footer {{ display: none !important; }}

/* Hide loading spinners */
.progress-bar, .progress-text, .meta-text, .meta-text-center,
div[class*="progress"], .eta-bar, .generating, .translucent {{
    display: none !important; opacity: 0 !important; height: 0 !important;
}}

/* Kill ALL borders, shadows, outlines, labels */
.block, .form, .wrap, .panel, .container,
[class*="block"], [class*="panel"] {{
    background: {BG} !important;
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
}}

/* Image components — seamless */
.image-container, .upload-container,
[data-testid="image"], .image-frame {{
    background: {BG} !important;
    border: none !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}}

/* Webcam placeholder — subtle, no dashed border */
.upload-container .wrap,
.upload-container .center {{
    background: {BG_CARD} !important;
    border: none !important;
    border-radius: 12px !important;
    color: {TEXT_MID} !important;
}}

/* Hide ALL labels, badges, icons on image components */
.label-wrap, span[data-testid="block-info"],
.icon-wrap, .icon-button, .download,
.image-container .label-wrap,
.image-container .icon-button,
.image-container .download {{
    display: none !important;
}}

/* Hide empty image placeholder icons */
.image-container .empty-image,
.image-container svg.empty {{
    display: none !important;
}}

/* Title */
.title h1 {{
    text-align: center;
    font-size: 1.6em;
    font-weight: 300;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: {TEXT};
    margin: 28px 0 4px 0;
}}
.sub {{ text-align: center; color: {TEXT_LIGHT}; font-size: 0.82em; margin: 0 0 20px 0; }}
.step p {{ text-align: center; color: {TEXT_MID}; font-size: 0.92em; margin: 4px 0; }}
.step strong {{ color: {TEXT} !important; }}
.status p {{ text-align: center; color: {ACCENT}; font-size: 0.85em; min-height: 1em; margin: 2px 0; }}

/* Loading animation */
@keyframes pulse {{
    0%, 100% {{ opacity: 0.4; }}
    50% {{ opacity: 1; }}
}}
.status p {{ animation: pulse 1.8s ease-in-out infinite; }}

/* Buttons */
.read-btn {{
    border: 1.5px solid {ACCENT} !important;
    background: transparent !important;
    color: {ACCENT} !important;
    border-radius: 8px !important;
    transition: all 0.2s !important;
}}
.read-btn:hover {{
    background: {ACCENT} !important;
    color: #fff !important;
}}
.reset-btn {{
    border: 1px solid {TEXT_LIGHT} !important;
    background: transparent !important;
    color: {TEXT_LIGHT} !important;
    border-radius: 8px !important;
}}
.reset-btn:hover {{
    border-color: {TEXT_MID} !important;
    color: {TEXT} !important;
}}

/* Fortune */
.fortune {{
    padding: 20px 8px;
    line-height: 1.8;
    color: {TEXT};
    font-size: 0.93em;
}}
.fortune blockquote {{
    border-left: 3px solid {ACCENT} !important;
    padding: 10px 18px !important;
    margin: 14px 0 !important;
    color: {TEXT} !important;
    font-style: italic !important;
    font-size: 1.05em !important;
    background: {LAVENDER} !important;
    border-radius: 0 8px 8px 0 !important;
}}
.fortune strong {{
    color: {ACCENT} !important;
}}
.fortune hr {{
    border: none !important;
    border-top: 1px solid {BG_CARD} !important;
    margin: 16px 0 !important;
}}

/* Markdown */
.markdown-text, .prose {{ color: {TEXT} !important; }}

/* Scrollbar */
::-webkit-scrollbar {{ width: 5px; }}
::-webkit-scrollbar-track {{ background: {BG}; }}
::-webkit-scrollbar-thumb {{ background: {BG_CARD}; border-radius: 3px; }}
"""

with gr.Blocks(
    theme=gr.themes.Base(
        primary_hue=gr.themes.colors.orange,
        neutral_hue=gr.themes.colors.stone,
        font=gr.themes.GoogleFont("Inter"),
    ),
    css=css,
    title="Hand Lines",
) as app:

    left_state = gr.State(None)
    right_state = gr.State(None)
    step_state = gr.State(0)

    gr.HTML('<div class="title"><h1>Hand Lines</h1></div>')
    gr.HTML('<div class="sub">Palm reading through computer vision</div>')

    step_text = gr.Markdown("Capture your **left palm**", elem_classes=["step"])

    webcam = gr.Image(sources=["webcam"], type="numpy", label="", mirror_webcam=True, height=320, show_label=False)

    status_text = gr.Markdown("", elem_classes=["status"])

    with gr.Row():
        read_btn = gr.Button("Read My Palm", size="lg", scale=3, elem_classes=["read-btn"], interactive=False)
        reset_btn = gr.Button("Reset", size="lg", scale=1, elem_classes=["reset-btn"])

    with gr.Row():
        left_output = gr.Image(type="numpy", label="Left", interactive=False, height=240, show_label=False)
        right_output = gr.Image(type="numpy", label="Right", interactive=False, height=240, show_label=False)

    fortune_output = gr.Markdown("", elem_classes=["fortune"])

    webcam.change(
        fn=on_capture,
        inputs=[webcam, step_state, left_state, right_state],
        outputs=[left_state, right_state, step_state, step_text, left_output, right_output, read_btn, status_text, webcam],
        show_progress="hidden",
    )

    read_btn.click(
        fn=on_read_start,
        outputs=[status_text, read_btn],
        show_progress="hidden",
    ).then(
        fn=on_read,
        inputs=[left_state, right_state],
        outputs=[left_output, right_output, fortune_output, status_text, read_btn],
        show_progress="hidden",
    )

    reset_btn.click(
        fn=on_reset,
        outputs=[left_state, right_state, step_state, step_text, left_output, right_output, read_btn, status_text, fortune_output, webcam],
        show_progress="hidden",
    )


if __name__ == "__main__":
    app.launch()
