from flask import Flask, jsonify
from flask_cors import CORS
import torch
import torchvision.models as models

app = Flask(__name__)
CORS(app)  # This is to allow cross-origin requests from your Svelte app

@app.route('/model-weights')
def get_model_weights():
    # Load the model (ensure the model architecture matches the weights)
    model = models.resnet50(pretrained=False)
    model.load_state_dict(torch.load('weights/vgg16_0_weights.pth', map_location=torch.device('cpu')))
    model.eval()
    print(model)

    # Extract weights for the first convolutional layer (as an example)
    # You might want to send more or process them depending on your needs
    conv1_weights = model.conv1.weight.data.numpy().tolist()

    # Send the weights as JSON
    return jsonify(conv1_weights)

if __name__ == '__main__':
    app.run(debug=True)