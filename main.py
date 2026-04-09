import cv2
import numpy as np
from fastapi import FastAPI

app = FastAPI()

# ---------------- GEO ----------------

def read_geo_file(filepath="geo.txt"):
    data = {}
    ordered_images = []

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

        ordered_images.append(name)

    return data, ordered_images


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


# ---------------- DETECTION BLEUE ----------------

def detect_blue_pipe(image_path):
    img = cv2.imread(f"images-2/{image_path}")

    if img is None:
        print(f"❌ Image not found: {image_path}")
        return None

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 🔵 bleu
    lower_blue = np.array([90, 100, 50])
    upper_blue = np.array([130, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # nettoyage
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # moments → centre du tube
    moments = cv2.moments(mask)

    if moments["m00"] == 0:
        print(f"❌ No blue detected: {image_path}")
        return None

    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])

    print(f"✅ {image_path} → {cx},{cy}")

    return [cx, cy]


# ---------------- MULTI POINTS ----------------

def compute_point(images_subset, geo_data):

    origins = []
    directions = []

    for img in images_subset:

        if img not in geo_data:
            continue

        pixel = detect_blue_pipe(img)
        if pixel is None:
            continue

        u, v = pixel

        cam = geo_data[img]
        C = cam["position"]
        yaw, pitch, roll = cam["orientation"]

        R = euler_to_rotation(yaw, pitch, roll)
        ray = R @ pixel_to_ray(u, v)

        origins.append(C)
        directions.append(ray)

    if len(origins) < 3:
        return None

    point = triangulate_rays(origins, directions)

    return point


# ---------------- API ----------------

@app.post("/reconstruct")
def reconstruct():

    geo_data, ordered_images = read_geo_file()

    points = []

    step = 5
    window = 10

    for i in range(0, len(ordered_images) - window, step):

        subset = ordered_images[i:i+window]

        point = compute_point(subset, geo_data)

        if point is not None:
            points.append(point.tolist())

    return {
        "points_3D": points,
        "count": len(points)
    }
