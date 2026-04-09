from fastapi import FastAPI
import numpy as np

app = FastAPI()

def read_geo_file(filepath="geo.txt"):
    data = {}

    with open(filepath, "r") as f:
        lines = f.readlines()[1:]  # skip EPSG

    for line in lines:
        parts = line.strip().split()
        name = parts[0]
        X, Y, Z = map(float, parts[1:4])

        data[name] = {
            "position": [X, Y, Z]
        }

    return data


@app.post("/triangulate")
def triangulate(data: dict):
    images = data["images"]

    geo_data = read_geo_file()

    observations = []

    for img in images:
        if img in geo_data:
            observations.append(geo_data[img])

    if not observations:
        return {"error": "No matching images found"}

    pts = [o["position"] for o in observations]
    point = np.mean(pts, axis=0)

    return {
        "X": float(point[0]),
        "Y": float(point[1]),
        "Z": float(point[2])
    }
