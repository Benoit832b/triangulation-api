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


# ---------------- DETECTION ----------------

def detect_red_pipe(image_path):
    img = cv2.imread(f"images/{image_path}")

    if img is None:
        return None

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170,120,70])
    upper_red2 = np.array([180,255,255])

    mask = cv2.inRange(hsv, lower_red1, upper_red1) + \
           cv2.inRange(hsv, lower_red2, upper_red2)

    edges = cv2.Canny(mask, 50, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180,
                            threshold=50,
                            minLineLength=100,
                            maxLineGap=20)

    if lines is None:
        return None

    best_line = None
    best_len = 0

    for line in lines:
        x1, y1, x2, y2 = line[0]

        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx*dx + dy*dy)

        if length > best_len:
            best_len = length
            best_line = (x1, y1, x2, y2)

    if best_line is None:
        return None

    x1, y1, x2, y2 = best_line

    cx = int((x1 + x2) / 2)
    cy = int(max(y1, y2))  # point bas

    return [cx, cy]


# ---------------- PIPELINE MULTI-POINTS ----------------

def compute_point(images_subset, geo_data):

    origins = []
    directions = []

    for img in images_subset:

        if img not in geo_data:
            continue

        pixel = detect_red_pipe(img)
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

    # 🔥 correction Z (fond tranchée)
    z_cam = np.mean([C[2] for C in origins])
    if point[2] > z_cam - 0.5:
        point[2] = z_cam - 1.2

    return point


# ---------------- API ----------------

@app.post("/reconstruct")
def reconstruct():

    geo_data, ordered_images = read_geo_file()

    points = []

    step = 5     # déplacement
    window = 10  # nb images

    for i in range(0, len(ordered_images) - window, step):

        subset = ordered_images[i:i+window]

        point = compute_point(subset, geo_data)

        if point is not None:
            points.append(point.tolist())

    # 🔥 lissage simple
    smoothed = []
    for i in range(1, len(points)-1):
        avg = (
            np.array(points[i-1]) +
            np.array(points[i]) +
            np.array(points[i+1])
        ) / 3
        smoothed.append(avg.tolist())

    return {
        "points_3D": smoothed,
        "count": len(smoothed)
    }
