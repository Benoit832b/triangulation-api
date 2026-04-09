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


@app.post("/triangulate")
def triangulate(data: dict):
    geo_data = read_geo_file()

    observations = data["observations"]

    pts = []

    for obs in observations:
        img = obs["image"]
        if img in geo_data:
            pts.append(geo_data[img]["position"])

    point = np.mean(pts, axis=0)

    return {
        "X": float(point[0]),
        "Y": float(point[1]),
        "Z": float(point[2])
    }
