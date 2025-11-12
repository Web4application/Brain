from flask import Flask, request, jsonify
from utils import load_model, save_model, append_data, load_new_data, mark_processed, train_incremental
from apscheduler.schedulers.background import BackgroundScheduler
import os, time, json

app = Flask(__name__)

os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("logs", exist_ok=True)

model = load_model()
last_train_time = 0
TRAIN_INTERVAL = 60  # seconds

def auto_train_incremental():
    global model, last_train_time
    new_data = load_new_data()
    if not new_data:
        print("[NeoMind] No new data to train.")
        return
    print(f"[NeoMind] Auto-training on {len(new_data)} new samples...")
    logs, metrics = train_incremental(model, new_data)
    save_model(model)
    mark_processed(new_data)
    last_train_time = time.time()
    print(f"[NeoMind] Auto-training done: {logs[-1]}")

scheduler = BackgroundScheduler()
scheduler.add_job(func=auto_train_incremental, trigger="interval", seconds=TRAIN_INTERVAL)
scheduler.start()

@app.route('/')
def home():
    return jsonify({"NeoMind": "Online", "status": "ready", "version": "v2 incremental"})

@app.route('/upload-data', methods=['POST'])
def upload_data_route():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400
    append_data(data)
    return jsonify({"status": "Data added", "entries": len(data)})

@app.route('/train', methods=['POST'])
def manual_train():
    auto_train_incremental()
    return jsonify({"status": "Manual incremental training triggered"})

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.get_json()
    if not input_data or "text" not in input_data:
        return jsonify({"error": "No text provided"}), 400
    vectorizer = model["vectorizer"]
    classifier = model["classifier"]
    label_encoder = model["label_encoder"]
    X = vectorizer.transform([input_data["text"]])
    y_pred = classifier.predict(X)
    label = label_encoder.inverse_transform(y_pred)[0]
    return jsonify({"prediction": label})

@app.route('/metrics', methods=['GET'])
def metrics():
    try:
        with open("logs/train_metrics.json", "r") as f:
            return jsonify(json.load(f))
    except:
        return jsonify({"accuracy": 0, "loss": 0, "message": "No training yet"})

@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        "model_loaded": True,
        "training_ready": True,
        "last_training": last_train_time
    })

if __name__ == '__main__':
    print("[NeoMind] Starting Incremental Live Learning Server...")
    app.run(host='0.0.0.0', port=5000, debug=True)
