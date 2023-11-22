from flask import Flask, jsonify, send_from_directory
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

@app.route("/rand")
def hello():
    return str(random.randint(0, 100))

if __name__ == "__main__":
    app.run(debug=True)
