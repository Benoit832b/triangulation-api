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
# ROTATION
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
# GEO
# =========================

def load_geo():

    data = {"observations": []}

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

    print("GEO loaded:", len(data["observations"]))

    return data


def get_camera_pose(geo, image_name):

    for obs in geo["observations"]:
        if obs["image"] == image_name:
            return np.array(obs["position"]), np.array(obs["rotation"])

    print("GEO not found:", image_name)
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
# DETECTION BLEU + DEBUG
# =========================

def detect_blue_pipe(image, debug_name="debug_mask.jpg"):

    try:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 🔵 seuil ajustable
        lower_blue = np.array([85, 70, 70])
        upper_blue = np.array([140, 255, 255])

        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # 🔥 sauvegarde debug
        cv2.imwrite(debug_name, mask)

        moments = cv2.moments(mask)

        if moments["m00"] == 0:
            print("❌ NO BLUE DETECTED")
            return None

        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])

        print(f"✅ DETECTED PIXEL: {cx}, {cy}")

        return [[cx, cy]]

    except Exception as e:
        print("detect error:", e)
        return None

# =========================
# TRIANGULATION
# =========================

def triangulate_points(P1, P2, pts1, pts2):

    try:
        pts1 = np.array(pts1).T.astype(np.float32)
        pts2 = np.array(pts2).T.astype(np.float32)

        if pts1.shape[1] < 1:
            return []

        points_4d = cv2.triangulatePoints(P1, P2, pts1, pts2)
        points_3d = points_4d[:3] / points_4d[3]

        return points_3d.T.tolist()

    except Exception as e:
        print("triangulation error:", e)
        return []

# =========================
# FILTRAGE
# =========================

def filter_points(points):

    filtered = []

    for p in points:
        try:
            x, y, z = p

            if 290 < z < 320:
                filtered.append(p)

        except:
            continue

    return filtered

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

        print("Images:", len(image_files))

        all_points = []

        for i in range(len(image_files) - 1):

            img1_name = image_files[i]
            img2_name = image_files[i + 1]

            print("Processing:", img1_name, img2_name)

            img1 = cv2.imread(f"images/{img1_name}")
            img2 = cv2.imread(f"images/{img2_name}")

            if img1 is None or img2 is None:
                print("❌ IMAGE READ ERROR")
                continue

            pts1 = detect_blue_pipe(img1, "debug_mask_1.jpg")
            pts2 = detect_blue_pipe(img2, "debug_mask_2.jpg")

            if pts1 is None or pts2 is None:
                print("❌ DETECTION FAILED")
                continue

            pos1, rot1 = get_camera_pose(geo, img1_name)
            pos2, rot2 = get_camera_pose(geo, img2_name)

            if pos1 is None or pos2 is None:
                print("❌ GEO ERROR")
                continue

            P1 = build_projection_matrix(pos1, rot1)
            P2 = build_projection_matrix(pos2, rot2)

            pts3d = triangulate_points(P1, P2, pts1, pts2)

            if len(pts3d) == 0:
                print("❌ TRIANGULATION FAILED")
                continue

            print("✅ POINT 3D:", pts3d)

            all_points.extend(pts3d)

        if len(all_points) == 0:
            return {"error": "no 3D points reconstructed"}

        filtered = filter_points(all_points)

        if len(filtered) == 0:
            return {"error": "no points after filtering"}

        return {
            "points_3D": filtered,
            "count": len(filtered)
        }

    except Exception as e:
        print("GLOBAL ERROR:", str(e))
        return {"error": str(e)}
