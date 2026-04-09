from fastapi import FastAPI
import numpy as np

app = FastAPI()

def triangulate_simple(obs):
    # moyenne des positions (version simplifiée pour démarrer)
    pts = [o["position"] for o in obs]
    return np.mean(pts, axis=0)

@app.post("/triangulate")
def triangulate(data: dict):
    point = triangulate_simple(data["observations"])
    return {
        "X": float(point[0]),
        "Y": float(point[1]),
        "Z": float(point[2])
    }