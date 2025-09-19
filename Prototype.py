import streamlit as st
import tempfile
import os
from PIL import Image
import io
import imageio.v2 as imageio
import numpy as np
import cv2
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops


canvas_size = (512, 512)
points_per_stroke = 200
num_inbetweens = 10


def order_points(coords):
    coords = coords.tolist()
    ordered = [coords.pop(0)]
    while coords:
        last = ordered[-1]
        dists = [np.linalg.norm(np.array(last)-np.array(c)) for c in coords]
        idx = int(np.argmin(dists))
        ordered.append(coords.pop(idx))
    return np.array(ordered)

def load_skeleton_strokes(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    _, img_bin = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY_INV)
    img_bin = cv2.dilate(img_bin, np.ones((2,2), np.uint8), iterations=1)
    skel = skeletonize(img_bin).astype(np.uint8)
    labeled = label(skel)
    strokes = []
    for region in regionprops(labeled):
        coords = region.coords
        if len(coords) < 2:
            continue
        coords_ordered = order_points(coords)
        strokes.append(coords_ordered)
    return strokes

def resample_points(points, n_points=200):
    diffs = np.diff(points, axis=0)
    dists = np.sqrt((diffs**2).sum(axis=1))
    cumdist = np.concatenate([[0], np.cumsum(dists)])
    if cumdist[-1] == 0:
        return np.repeat(points[0:1], n_points, axis=0)
    new_dist = np.linspace(0, cumdist[-1], n_points)
    x_new = np.interp(new_dist, cumdist, points[:,1])
    y_new = np.interp(new_dist, cumdist, points[:,0])
    return np.stack([y_new, x_new], axis=1)

def interpolate_strokes(strokes1, strokes2, alpha):
    frames = []
    n_strokes = min(len(strokes1), len(strokes2))
    for i in range(n_strokes):
        p1 = resample_points(strokes1[i], points_per_stroke)
        p2 = resample_points(strokes2[i], points_per_stroke)
        interp = (1-alpha)*p1 + alpha*p2
        frames.append(interp)
    return frames

def draw_strokes(strokes, size):
    img = Image.new("RGB", size, "white")
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    for stroke in strokes:
        for i in range(1, len(stroke)):
            y1, x1 = stroke[i-1]
            y2, x2 = stroke[i]
            draw.line([x1, y1, x2, y2], fill="black", width=2)
    return img

def generate_inbetweens(img1_path, img2_path, n_inbetweens=10):
    strokes1 = load_skeleton_strokes(img1_path)
    strokes2 = load_skeleton_strokes(img2_path)
    frames = []
    frames.append(draw_strokes(strokes1, canvas_size))
    for i in range(1, n_inbetweens+1):
        alpha = i / (n_inbetweens + 1)
        interp_strokes = interpolate_strokes(strokes1, strokes2, alpha)
        frame_img = draw_strokes(interp_strokes, canvas_size)
        frames.append(frame_img)
    frames.append(draw_strokes(strokes2, canvas_size))
    return frames

def make_sprite_sheet(frames):
    sheet_width = canvas_size[0] * len(frames)
    sheet_height = canvas_size[1]
    sprite_sheet = Image.new("RGB", (sheet_width, sheet_height), "white")
    for idx, f in enumerate(frames):
        sprite_sheet.paste(f, (idx*canvas_size[0], 0))
    return sprite_sheet

def make_gif(frames):
    gif_bytes = io.BytesIO()
    frames[0].save(gif_bytes, format="GIF", save_all=True,
                   append_images=frames[1:], duration=100, loop=0)
    gif_bytes.seek(0)
    return gif_bytes


st.title("🎬 FrameFlow Prototype: AI-Powered Stroke Inbetweening")

uploaded1 = st.file_uploader("Upload first keyframe", type=["png","jpg","jpeg"])
uploaded2 = st.file_uploader("Upload second keyframe", type=["png","jpg","jpeg"])

if uploaded1 and uploaded2:
    temp1 = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    temp1.write(uploaded1.read())
    temp1.close()

    temp2 = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    temp2.write(uploaded2.read())
    temp2.close()

    if st.button("Generate Inbetweens"):
        with st.spinner("Generating..."):
            frames = generate_inbetweens(temp1.name, temp2.name, num_inbetweens)
            sprite_sheet = make_sprite_sheet(frames)
            gif_bytes = make_gif(frames)

        st.success("Done! 🎉")

        # Show sprite sheet
        st.subheader("Sprite Sheet")
        st.image(sprite_sheet, use_container_width=True)

        # Show gif
        st.subheader("Animated GIF Preview")
        st.image(gif_bytes)

        # Downloads
        st.download_button("Download Sprite Sheet", data=sprite_sheet.tobytes(),
                           file_name="sprite_sheet.png", mime="image/png")
        st.download_button("Download GIF", data=gif_bytes,
                         file_name="preview.gif", mime="image/gif")
