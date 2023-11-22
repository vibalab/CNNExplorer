import torch
import torchvision.models as models
import json
import os

def convert_pth_to_json(pth_file_path, json_file_path):
    if 'resnet50' in pth_file_path:
        # Load the pretrained ResNet50 model
        model = models.resnet50(weights=None)
    elif 'vgg16' in pth_file_path:
        return
        model = models.vgg16(weights=None)
    elif 'resnet18' in pth_file_path:
        return
        model = models.resnet18(weights=None)  # Initialize a blank ResNet18 model
    elif 'googlenet' in pth_file_path:
        model = models.googlenet(weights=None)  # Initialize a blank googlenet model

    # Load the weights from the .pth file
    # Replace 'path_to_resnet50_0_weights.pth' with the actual file path
    model.load_state_dict(torch.load(pth_file_path, map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode

    # Print the model structure
    print(model)

    # Print additional information such as layer names and parameter sizes
    print("\nModel's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    print("\nNumber of parameters:", sum(p.numel() for p in model.parameters()))

    # Extract weights and biases from each layer and convert them to a Python dictionary
    model_weights = {name: param.tolist() for name, param in model.state_dict().items()}

    # Serialize the dictionary to a JSON file
    with open(json_file_path, 'w') as f:
        json.dump(model_weights, f)

    print(f"Model weights {pth_file_path} have been converted to {json_file_path}")


if __name__ == "__main__":
    weight_dir = 'weights/'
    for weight_f in os.listdir(weight_dir):
        if weight_f.endswith('.pth'):
            convert_pth_to_json(weight_dir + weight_f, weight_dir + weight_f[:-4]+'.json')