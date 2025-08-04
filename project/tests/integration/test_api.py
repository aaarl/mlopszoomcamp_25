import requests

def test_api_predict():
    sample = {
        "radius_mean": 14.5,
        "texture_mean": 20.1,
        "perimeter_mean": 90.2,
        "area_mean": 550.0,
        "smoothness_mean": 0.1,
        "compactness_mean": 0.12,
        "concavity_mean": 0.13,
        "concave_points_mean": 0.14
    }

    res = requests.post("http://localhost:8000/predict", json=sample)
    assert res.status_code == 200
    assert "prediction" in res.json()
