import cv2
import numpy as np
from fastapi import FastAPI
import json
import os

app = FastAPI()

# =========================
# PARAMÈTRES
# =========================

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

# caméra approx (à affiner plus tard)
FOCAL = 1000
CX = IMAGE_WIDTH / 2
CY = IMAGE_HEIGHT / 2

# =========================
# UTILS
# =========================

def load_geo():
    with open("geo.txt", "r") as f:
        return json.load(f)

def get_camera_pose(geo, image_name):
    for obs in geo["observations"]:
        if obs["image"] == image_name:
            return np.array(obs["position"]), np.array(obs["rotation"])
    return None, None

def build_projection_matrix(position, rotation):
    R = np.array(rotation)
    t = -R @ position

    K = np.array([
        [FOCAL, 0, CX],
        [0, FOCAL, CY],
        [0, 0, 1]
    ])

    RT = np.hstack((R, t.reshape(3, 1)))
    return K @ RT

# =========================
# DÉTECTION FOURREAU BLEU
# =========================

def detect_blue_pipe(image):

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # plage BLEU robuste terrain
    lower_blue = np.array([90, 80, 80])
    upper_blue = np.array([130, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # nettoyage
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

    # contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # prendre le plus grand contour (le fourreau)
    cnt = max(contours, key=cv2.contourArea)

    if cv2.contourArea(cnt) < 500:
        return None

    # approximation ligne (PCA)
    data_pts = cnt.reshape(-1, 2).astype(np.float32)
    mean, eigenvectors = cv2.PCACompute(data_pts, mean=None)

    center = mean[0]
    direction = eigenvectors[0]

    # générer points le long du fourreau
    points = []
    for t in np.linspace(-200, 200, 10):  # 10 points par image
        pt = center + t * direction
        x, y = int(pt[0]), int(pt[1])

        if 0 <= x < IMAGE_WIDTH and 0 <= y < IMAGE_HEIGHT:
            points.append([x, y])

    return points

# =========================
# TRIANGULATION
# =========================

def triangulate_points(P1, P2, pts1, pts2):

    pts1 = np.array(pts1).T.astype(np.float32)
    pts2 = np.array(pts2).T.astype(np.float32)

    points_4d = cv2.triangulatePoints(P1, P2, pts1, pts2)
    points_3d = points_4d[:3] / points_4d[3]

    return points_3d.T

# =========================
# FILTRAGE
# =========================

def filter_points(points):

    filtered = []

    for p in points:
        x, y, z = p

        # contraintes terrain
        if 300 < z < 315:   # à adapter selon chantier
            filtered.append(p)

    return filtered

# =========================
# INTERPOLATION 10cm
# =========================

def interpolate_polyline(points):

    if len(points) < 2:
        return points

    result = [points[0]]

    for i in range(len(points) - 1):
        p1 = np.array(points[i])
        p2 = np.array(points[i+1])

        dist = np.linalg.norm(p2 - p1)

        steps = max(1, int(dist / 0.1))  # 10 cm

        for s in range(1, steps):
            pt = p1 + (p2 - p1) * (s / steps)
            result.append(pt.tolist())

        result.append(p2.tolist())

    return result

# =========================
# API
# =========================

@app.get("/reconstruct")
def reconstruct():

    geo = load_geo()

    image_files = sorted(os.listdir("images"))

    all_points = []

    for i in range(len(image_files) - 1):

        img1_name = image_files[i]
        img2_name = image_files[i+1]

        img1 = cv2.imread(f"images/{img1_name}")
        img2 = cv2.imread(f"images/{img2_name}")

        if img1 is None or img2 is None:
            continue

        pts1 = detect_blue_pipe(img1)
        pts2 = detect_blue_pipe(img2)

        if pts1 is None or pts2 is None:
            continue

        pos1, rot1 = get_camera_pose(geo, img1_name)
        pos2, rot2 = get_camera_pose(geo, img2_name)

        if pos1 is None or pos2 is None:
            continue

        P1 = build_projection_matrix(pos1, rot1)
        P2 = build_projection_matrix(pos2, rot2)

        # matcher les points par index
        n = min(len(pts1), len(pts2))
        pts3d = triangulate_points(P1, P2, pts1[:n], pts2[:n])

        all_points.extend(pts3d.tolist())

    # filtrage
    filtered = filter_points(all_points)

    # tri spatial (important pour ligne)
    filtered.sort(key=lambda p: p[0])

    # interpolation 10 cm
    dense = interpolate_polyline(filtered)

    return {
        "points_3D": dense,
        "count": len(dense)
    }
