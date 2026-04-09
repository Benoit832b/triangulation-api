from fastapi import FastAPI
import numpy as np

app = FastAPI()

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


def pixel_to_ray(u, v):
    # intrinsics simplifiées
    fx = fy = 1000
    cx = 640
    cy = 360

    x = (u - cx) / fx
    y = (v - cy) / fy

    ray = np.array([x, y, 1.0])
    return ray / np.linalg.norm(ray)


def triangulate_rays(origins, directions):
    A = []
    b = []

    for C, d in zip(origins, directions):
        I = np.eye(3)
        A.append(I - np.outer(d, d))
        b.append((I - np.outer(d, d)) @ C)

    A = np.vstack(A)
    b = np.vstack(b)

    X, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return X.flatten()


@app.post("/triangulate")
def triangulate(data: dict):
    geo_data = read_geo_file()

    origins = []
    directions = []

    for obs in data["observations"]:
        img = obs["image"]
        u, v = obs["pixel"]

        if img not in geo_data:
            continue

        cam = geo_data[img]
        C = cam["position"]
        yaw, pitch, roll = cam["orientation"]

        R = euler_to_rotation(yaw, pitch, roll)
        ray_cam = pixel_to_ray(u, v)
        ray_world = R @ ray_cam

        origins.append(C)
        directions.append(ray_world)

    point = triangulate_rays(origins, directions)

    return {
        "X": float(point[0]),
        "Y": float(point[1]),
        "Z": float(point[2])
    }
