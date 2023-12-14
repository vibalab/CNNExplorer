from flask import Flask, jsonify, send_from_directory
import torch
import torchvision.models as models
import json
import random

app = Flask(__name__)

@app.route("/")
def base():
    return send_from_directory('svelte-app/public', 'index.html')

@app.route('/<path:path>')
def home(path):
    return send_from_directory('svelte-app/public', path)

@app.route('/layer')
def get_layer_output():
    with open(f'layer_outputs/layer_1_batch_0.json', 'r') as f:
        data = json.load(f)
    return jsonify(data)

@app.route('/weight')
def get_wegiht():
    model = models.resnet18(weights=None)
    model.load_state_dict(torch.load('./weights/resnet18_0_weights.pth', map_location=torch.device('cpu')))
    
    # layer_names = []
    # for name, _ in model.named_parameters():
    #     layer_names.append(name)
    
    first_conv_weights = model.conv1.weight.data.cpu().numpy()
    weights_dict = {'conv1_weights': first_conv_weights.tolist()}
    
    return jsonify(weights_dict)

@app.route("/rand")
def hello():
    return str(random.randint(0, 100))

if __name__ == "__main__":
    app.run(debug=True)
