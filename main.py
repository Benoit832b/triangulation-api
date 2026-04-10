import cv2
import numpy as np
import os
import base64
import urllib.request
import torch
import time
from fastapi import FastAPI
from segment_anything import sam_model_registry, SamPredictor

app = FastAPI()

# =========================
# 📥 SAM DOWNLOAD
# =========================

MODEL_PATH = "/tmp/sam_vit_b_01ec64.pth"
MODEL_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"

if not os.path.exists(MODEL_PATH):
    print("📥 Downloading SAM model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("✅ SAM model downloaded")

# =========================
# 🧠 LOAD SAM
# =========================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry["vit_b"](checkpoint=MODEL_PATH)
sam.to(device=DEVICE)

predictor = SamPredictor(sam)

print("✅ SAM READY on", DEVICE)

# =========================
# 📷 CAMERA PARAMS
# =========================

FOCAL = 1000
CX = 640
CY = 360

# =========================
# 🔄 ROTATION
# =========================

def euler_to_rotation(yaw, pitch, roll):
    yaw = np.radians(yaw)
    pitch = np.radians(pitch)
    roll = np.radians(roll)

    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0, 0, 1]
    ])

    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll),  np.cos(roll)]
    ])

    return Rz @ Ry @ Rx

# =========================
# 📄 GEO
# =========================

def load_geo():
    observations = []

    with open("geo.txt", "r") as f:
        lines = f.readlines()

    for line in lines[1:]:
        parts = line.strip().split()

        if len(parts) < 7:
            continue

        name = parts[0]
        X, Y, Z = map(float, parts[1:4])
        yaw, pitch, roll = map(float, parts[4:7])

        R = euler_to_rotation(yaw, pitch, roll)

        observations.append({
            "image": name,
            "position": np.array([X, Y, Z]),
            "rotation": R
        })

    return observations


def get_pose(geo, name):
    for obs in geo:
        if obs["image"] == name:
            return obs["position"], obs["rotation"]
    return None, None

# =========================
# 📐 PROJECTION
# =========================

def build_projection(position, rotation):

    t = -rotation @ position

    K = np.array([
        [FOCAL, 0, CX],
        [0, FOCAL, CY],
        [0, 0, 1]
    ])

    RT = np.hstack((rotation, t.reshape(3, 1)))

    return K @ RT

# =========================
# 🤖 SAM DETECTION (ULTRA SIMPLE)
# =========================

def detect_pipe_sam(image):

    predictor.set_image(image)

    h, w = image.shape[:2]

    input_point = np.array([[w // 2, h // 2]])
    input_label = np.array([1])

    masks, scores, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False
    )

    mask = (masks[0] * 255).astype(np.uint8)

    moments = cv2.moments(mask)

    if moments["m00"] < 500:
        return None

    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])

    return [[cx, cy]]

# =========================
# 🔺 TRIANGULATION
# =========================

def triangulate(P1, P2, pts1, pts2):

    pts1 = np.array(pts1).T.astype(np.float32)
    pts2 = np.array(pts2).T.astype(np.float32)

    pts4d = cv2.triangulatePoints(P1, P2, pts1, pts2)
    pts3d = pts4d[:3] / pts4d[3]

    return pts3d.T.tolist()

# =========================
# 🌐 API
# =========================

@app.get("/reconstruct")
def reconstruct():

    start_time = time.time()

    geo = load_geo()

    images = sorted(os.listdir("images"))

    # 🔥 ULTRA LIMITATION
    if len(images) < 2:
        return {"error": "not enough images"}

    img1_name = images[0]
    img2_name = images[1]

    img1 = cv2.imread(f"images/{img1_name}")
    img2 = cv2.imread(f"images/{img2_name}")

    pts1 = detect_pipe_sam(img1)
    pts2 = detect_pipe_sam(img2)

    if pts1 is None or pts2 is None:
        return {"error": "SAM detection failed"}

    pos1, rot1 = get_pose(geo, img1_name)
    pos2, rot2 = get_pose(geo, img2_name)

    if pos1 is None or pos2 is None:
        return {"error": "geo mismatch"}

    P1 = build_projection(pos1, rot1)
    P2 = build_projection(pos2, rot2)

    pts3d = triangulate(P1, P2, pts1, pts2)

    duration = time.time() - start_time

    return {
        "points_3D": pts3d,
        "count": len(pts3d),
        "processing_time_sec": duration
    }
