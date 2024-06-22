import os
import threading
from queue import Queue
from flask import Flask, request, jsonify
from transformers import AutoModelForImageClassification
import timm
import torch
from PIL import Image
from torchvision import transforms

app = Flask(__name__)
models = {}
model_locks = {}
UPLOAD_FOLDER = 'uploaded_images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Determine number of GPUs and create a queue for inference tasks
num_gpus = torch.cuda.device_count()
inference_queue = Queue()

def load_model(model_name):
    try:
        if model_name.startswith('timm/'):
            model = timm.create_model(model_name.replace('timm/', ''), pretrained=True)
        else:
            model = AutoModelForImageClassification.from_pretrained(model_name)
        models[model_name] = model
        print(f"Model '{model_name}' loaded successfully.")
    except Exception as e:
        print(f"Failed to load model '{model_name}': {str(e)}")
        models[model_name] = None

@app.route('/load_model', methods=['GET'])
def load_model_endpoint():
    model_name = request.args.get('model_name')
    if not model_name:
        return jsonify({"error": "model_name parameter is required"}), 400

    if model_name in models:
        return jsonify({"message": f"Model '{model_name}' is already loaded."})

    if model_name not in model_locks:
        model_locks[model_name] = threading.Lock()

    def load_model_thread():
        with model_locks[model_name]:
            if model_name not in models:
                load_model(model_name)

    thread = threading.Thread(target=load_model_thread)
    thread.start()

    return jsonify({"message": f"Model '{model_name}' is being loaded."})

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image = request.files['image']
    if image.filename == '':
        return jsonify({"error": "No selected file"}), 400

    image_path = os.path.join(UPLOAD_FOLDER, image.filename)
    image.save(image_path)
    return jsonify({"message": f"Image '{image.filename}' uploaded successfully", "path": image_path})

def preprocess_image(image_path, model_name):
    input_image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(input_image).unsqueeze(0)

def inference_worker():
    while True:
        model_name, image_path, result_queue = inference_queue.get()
        try:
            result = run_inference(model_name, image_path)
            result_queue.put(result)
        except Exception as e:
            result_queue.put(f"Failed to run inference: {str(e)}")
        inference_queue.task_done()

def run_inference(model_name, image_path):
    if model_name not in models or not models[model_name]:
        raise ValueError("Model not properly loaded")

    model = models[model_name]
    input_tensor = preprocess_image(image_path, model_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        model.eval()
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)

    model.to('cpu')
    torch.cuda.empty_cache()

    return predicted.item()

@app.route('/infer', methods=['POST'])
def infer():
    model_name = request.args.get('model_name')
    image_path = request.args.get('image_path')

    if not model_name:
        return jsonify({"error": "model_name parameter is required"}), 400
    if not image_path:
        return jsonify({"error": "image_path parameter is required"}), 400
    if model_name not in models:
        return jsonify({"error": f"Model '{model_name}' is not loaded"}), 400
    if not os.path.exists(image_path):
        return jsonify({"error": f"Image '{image_path}' does not exist"}), 400

    result_queue = Queue()
    inference_queue.put((model_name, image_path, result_queue))

    result = result_queue.get()  # Wait for the inference result

    return jsonify({"message": "Inference completed", "result": result})

# Start inference worker threads
for _ in range(num_gpus):
    worker = threading.Thread(target=inference_worker)
    worker.daemon = True
    worker.start()

if __name__ == '__main__':
    app.run(debug=True)
