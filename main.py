import cv2
import numpy as np
from fastapi import FastAPI

app = FastAPI()

# ---------------- GEO ----------------

def read_geo_file(filepath="geo.txt"):
    data = {}

    with open(filepath, "r") as f:
        lines = f.readlines()[1:]

    for line in lines:
        parts = line.strip().split()
        name = parts[0]

        X, Y, Z = map(float, parts[1:4])
        yaw, pitch, roll = map(float, parts[4:7])

        data[name] = {
            "position": np.array([X, Y, Z]),
            "orientation": [yaw, pitch, roll]
        }

    return data


# ---------------- ROTATION ----------------

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


# ---------------- CAMERA ----------------

def pixel_to_ray(u, v):
    # intrinsics réalistes pour 1280x720
    fx = fy = 900
    cx = 640
    cy = 360

    x = (u - cx) / fx
    y = (v - cy) / fy

    ray = np.array([x, y, 1.0])
    return ray / np.linalg.norm(ray)


# ---------------- TRIANGULATION ----------------

def triangulate_rays(origins, directions):
    A = []
    b = []

    for C, d in zip(origins, directions):
        d = d / np.linalg.norm(d)

        I = np.eye(3)
        M = I - np.outer(d, d)

        A.append(M)
        b.append(M @ C)

    A = np.concatenate(A, axis=0)
    b = np.concatenate(b, axis=0)

    X, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    return X


# ---------------- DETECTION LIGNE ----------------

def detect_red_pipe(image_path):
    img = cv2.imread(image_path)

    if img is None:
        print(f"Image not found: {image_path}")
        return None

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # masque rouge
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170,120,70])
    upper_red2 = np.array([180,255,255])

    mask = cv2.inRange(hsv, lower_red1, upper_red1) + \
           cv2.inRange(hsv, lower_red2, upper_red2)

    # edge detection
    edges = cv2.Canny(mask, 50, 150)

    # Hough transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                           minLineLength=100, maxLineGap=20)

    if lines is None:
        return None

    best_line = None
    best_score = 0

    for line in lines:
        x1, y1, x2, y2 = line[0]

        dx = x2 - x1
        dy = y2 - y1

        length = np.sqrt(dx*dx + dy*dy)

        if length < 100:
            continue

        angle = abs(dy) / (abs(dx) + 1e-5)

        # 🔥 filtre : lignes plutôt verticales (fourreau)
        if angle < 1:
            continue

        score = length

        if score > best_score:
            best_score = score
            best_line = (x1, y1, x2, y2)

    if best_line is None:
        return None

    x1, y1, x2, y2 = best_line

    # centre de la ligne
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)

    return [cx, cy]


# ---------------- API ----------------

@app.post("/triangulate")
def triangulate(data: dict):
    geo_data = read_geo_file()

    origins = []
    directions = []

    for obs in data["observations"]:
        img = obs["image"]

        if img not in geo_data:
            continue

        cam = geo_data[img]
        C = cam["position"]
        yaw, pitch, roll = cam["orientation"]

        # pixel manuel ou auto
        if "pixel" in obs:
            u, v = obs["pixel"]
        else:
            pixel = detect_red_pipe(img)
            if pixel is None:
                print(f"No detection for {img}")
                continue
            u, v = pixel

        R = euler_to_rotation(yaw, pitch, roll)
        ray_cam = pixel_to_ray(u, v)
        ray_world = R @ ray_cam

        origins.append(C)
        directions.append(ray_world)

    if len(origins) < 2:
        return {"error": "Not enough valid observations"}

    try:
        point = triangulate_rays(origins, directions)

        # 🔥 CONTRAINTE SOL (Z réaliste)
        avg_cam_z = np.mean([c[2] for c in origins])

        if abs(point[2] - avg_cam_z) > 5:
            return {"error": "Z out of realistic range"}

        # 🔥 REJET DES POINTS ABERRANTS
        dist = np.linalg.norm(point - origins[0])
        if dist > 50:
            return {"error": "Triangulation unstable (too far)"}

        return {
            "X": float(point[0]),
            "Y": float(point[1]),
            "Z": float(point[2])
        }

    except Exception as e:
        return {
            "error": "Triangulation failed",
            "details": str(e)
        }
