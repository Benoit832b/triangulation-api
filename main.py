import cv2
import numpy as np
from fastapi import FastAPI
import os

app = FastAPI()

# =========================
# PARAMÈTRES
# =========================

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

FOCAL = 1000
CX = IMAGE_WIDTH / 2
CY = IMAGE_HEIGHT / 2

# =========================
# ROTATION (CRITIQUE)
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
# GEO (FORMAT TEXTE)
# =========================

def load_geo():
    data = {"observations": []}

    try:
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

            data["observations"].append({
                "image": name,
                "position": [X, Y, Z],
                "rotation": R.tolist()
            })

        print(f"📍 GEO LOADED: {len(data['observations'])}")

        return data

    except Exception as e:
        raise Exception(f"geo.txt parsing error: {str(e)}")


def get_camera_pose(geo, image_name):
    for obs in geo["observations"]:
        if obs["image"] == image_name:
            return np.array(obs["position"]), np.array(obs["rotation"])

    print(f"❌ GEO NOT FOUND: {image_name}")
    return None, None

# =========================
# PROJECTION
# =========================

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
# DETECTION BLEUE
# =========================

def detect_blue_pipe(image):

    try:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_blue = np.array([85, 70, 70])
        upper_blue = np.array([140, 255, 255])

        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        cnt = max(contours, key=cv2.contourArea)

        if cv2.contourArea(cnt) < 500:
            return None

        data_pts = cnt.reshape(-1, 2).astype(np.float32)

        if len(data_pts) < 10:
            return None

        mean, eigenvectors = cv2.PCACompute(data_pts, mean=None)

        center = mean[0]
        direction = eigenvectors[0]

        points = []

        for t in np.linspace(-200, 200, 10):
            pt = center + t * direction
            x, y = int(pt[0]), int(pt[1])

            if 0 <= x < IMAGE_WIDTH and 0 <= y < IMAGE_HEIGHT:
                points.append([x, y])

        if len(points) < 2:
            return None

        return points

    except Exception as e:
        print("❌ detect_blue_pipe error:", e)
        return None

# =========================
# TRIANGULATION
# =========================

def triangulate_points(P1, P2, pts1, pts2):

    try:
        pts1 = np.array(pts1).T.astype(np.float32)
        pts2 = np.array(pts2).T.astype(np.float32)

        if pts1.shape[1] < 2:
            return []

        points_4d = cv2.triangulatePoints(P1, P2, pts1, pts2)
        points_3d = points_4d[:3] / points_4d[3]

        return points_3d.T.tolist()

    except Exception as e:
        print("❌ triangulation error:", e)
        return []

# =========================
# FILTRAGE
# =========================

def filter_points(points):

    filtered = []

    for p in points:
        try:
            x, y, z = p

            if 295 < z < 320:  # élargi volontairement
                filtered.append(p)

        except:
            continue

    return filtered

# =========================
# INTERPOLATION
# =========================

def interpolate_polyline(points):

    if len(points) < 2:
        return points

    result = [points[0]]

    for i in range(len(points) - 1):

        p1 = np.array(points[i])
        p2 = np.array(points[i + 1])

        dist = np.linalg.norm(p2 - p1)

        steps = max(1, int(dist / 0.1))

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

    try:

        geo = load_geo()

        if not os.path.exists("images"):
            return {"error": "images folder not found"}

        image_files = sorted(os.listdir("images"))

        if len(image_files) < 2:
            return {"error": "not enough images"}

        print(f"📸 {len(image_files)} images")

        all_points = []

        for i in range(len(image_files) - 1):

            img1_name = image_files[i]
            img2_name = image_files[i + 1]

            print(f"➡️ {img1_name} / {img2_name}")

            img1 = cv2.imread(f"images/{img1_name}")
            img2 = cv2.imread(f"images/{img2_name}")

            if img1 is None or img2 is None:
                print("❌ image read error")
                continue

            pts1 = detect_blue_pipe(img1)
            pts2 = detect_blue_pipe(img2)

            if pts1 is None or pts2 is None:
                print("❌ detection failed")
                continue

            pos1, rot1 = get_camera_pose(geo, img1_name)
            pos2, rot2 = get_camera_pose(geo, img2_name)

            if pos1 is None or pos2 is None:
                print("❌ geo mismatch")
                continue

            P1 = build_projection_matrix(pos1, rot1)
            P2 = build_projection_matrix(pos2, rot2)

            pts3d = triangulate_points(P1, P2, pts1, pts2)

            if len(pts3d) == 0:
                print("❌ triangulation failed")
                continue

            all_points.extend(pts3d)

        print(f"🔵 points bruts: {len(all_points)}")

        if len(all_points) == 0:
            return {"error": "no 3D points reconstructed"}

        filtered = filter_points(all_points)

        print(f"🟢 points filtrés: {len(filtered)}")

        if len(filtered) == 0:
            return {"error": "no points after filtering"}

        filtered.sort(key=lambda p: p[0])

        dense = interpolate_polyline(filtered)

        print(f"📏 points densifiés: {len(dense)}")

        return {
            "points_3D": dense,
            "count": len(dense)
        }

    except Exception as e:
        print("🔥 GLOBAL ERROR:", str(e))
        return {"error": str(e)}
