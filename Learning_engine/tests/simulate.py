import requests

URL = "http://localhost:5000"

# Simulated sensor / text data
data_samples = [
    {"text": "NeoMind detects obstacle", "label": "obstacle"},
    {"text": "NeoMind identifies target", "label": "target"},
    {"text": "NeoMind standby mode", "label": "standby"}
]

# Upload data
resp = requests.post(f"{URL}/upload-data", json=data_samples)
print("Upload response:", resp.json())

# Trigger training
resp = requests.post(f"{URL}/train")
print("Training response:", resp.json())

# Test prediction
test_text = {"text": "NeoMind detects obstacle"}
resp = requests.post(f"{URL}/predict", json=test_text)
print("Prediction response:", resp.json())
