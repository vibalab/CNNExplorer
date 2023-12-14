from flask import Flask, jsonify, send_from_directory
import json
import random
import torch
import os

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

@app.route('/<string:model>', methods=['GET'])
def get_model(model):
    directory = f'./svelte-app/public/output/'
    filename = os.path.join(directory, f'{model}_info.json')
    with open(filename, 'r') as f:
        data = json.load(f)

    layer = request.args.get('layer', None)
    key = request.args.get('key', None)
    
    if layer is None and key is None:
        return jsonify(data)
    elif layer is not None and key is None:
        return jsonify(data[layer])
    elif layer is not None and key is not None:
        if key in ['input', 'output', 'weight']:
            return jsonify(data[layer][key])
        else:
            raise ValueError("key must be in ['input, 'output', 'weight']")
    else:
        raise KeyError("layer must be defined to reference key")

@app.route("/rand")
def hello():
    return str(random.randint(0, 100))

if __name__ == "__main__":
    app.run(debug=True)
