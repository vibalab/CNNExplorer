import os
import torch
import json

def convert_pt_to_json(pt_file, json_file):
    if torch.cuda.is_available():
        # Load the PyTorch tensor
        tensor_data = torch.load(pt_file)
    else:
        tensor_data = torch.load(pt_file, map_location=torch.device('cpu'))
    
    # Convert to a list (if necessary, depending on your needs)
    tensor_list = tensor_data.tolist()
    
    # Save to a JSON file
    with open(json_file, 'w') as f:
        json.dump(tensor_list, f)


if __name__ == "__main__":
    weight_dir = 'layer_outputs/'
    for weight_f in os.listdir(weight_dir):
        if weight_f.endswith('.pt'):
            convert_pt_to_json(weight_dir + weight_f, weight_dir + weight_f[:-3]+'.json')

# convert_pt_to_json('layer_output.pt', 'layer_output.json')