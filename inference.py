import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import json
import os

from convert_weight_to_json import *
from convert_tensor_to_json import *

layer_inputs = []
layer_outputs = []
layer_names = []
layer_classes = []

check_list_class = [nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d, nn.AdaptiveAvgPool2d, nn.Linear]

def hook_fn(m, i, o):
    layer_names.append(m._layer_name)
    layer_classes.append(m.__class__)
    layer_inputs.append(i)
    layer_outputs.append(o)

def check_layer(name, module):
    if not hasattr(module, "_modules"):
        return True
    if isinstance(module, nn.Sequential):
        return False
    for c in check_list_class:
        if isinstance(module, c):
             return True
    return False

# when accessing same module twice?
def hook_all_layers(net, name=""):
    split = "" if name == "" else "."
    for lname, layer in net._modules.items():
        #print(lname, check_layer(lname, layer))
        if not check_layer(lname, layer):
            hook_all_layers(layer, name=f"{name}{split}{lname}")
        else:
            layer.register_forward_hook(hook_fn)
            setattr(layer, "_layer_name" ,f"{name}{split}{lname}")

def load_pretrained_model(model_name, numbering):
    if model_name == "googlenet":
        model = models.googlenet(weights=None, aux_logits=False)
    elif model_name == "resnet18":
        model = models.resnet18(weights=None)
    elif model_name == "alexnet":
        model = models.alexnet(weights=None)
    elif model_name == "vgg16":
        model = models.vgg16(weights=None)
    else:
        print("Unsupported model name!")
        return None

    weight_file = f"weights/{model_name}_{numbering}_weights.pth"
    model.load_state_dict(torch.load(weight_file))
    model.eval()

    hook_all_layers(model)
    
    return model

def get_layer_state(state_dict, layer_name):
    result = dict()
    for k in state_dict.keys():
        if k.startswith(layer_name):
            result[k] = state_dict[k]
    return result

def inference(log_dir, data_dir, model_name): 
    os.makedirs(log_dir, exist_ok=True)

    model = load_pretrained_model(model_name, 0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    data = torchvision.io.read_image(data_dir).float()
    data = data.unsqueeze(0)

    info = {}

    with torch.no_grad():
        data = data.to(device)
        layer_inputs.clear()
        layer_outputs.clear()
        outputs = model(data)
        state_dict = model.state_dict()
        for i, o, n, c in zip(layer_inputs, layer_outputs, layer_names, layer_classes):
            i = i[0].squeeze(0).cpu().numpy()
            o = o[0].cpu().numpy()

            layer_state = get_layer_state(state_dict, n)
            layer_weight = None
            if n+".weight" in layer_state:
                layer_weight = layer_state[n+".weight"].cpu().numpy()
                print(n, " : ", i.shape, o.shape, layer_weight.shape)
            else:
                print(n, " : ", i.shape, o.shape)

            # reduce dimension
            input_idx = None
            output_idx = None
            if i.shape[0] > 8:
                input_idx = range(8)
                i = i[input_idx]
            if o.shape[0] > 8: 
                output_idx = range(8)
                o = o[output_idx]
            
            # reduce dimension (weight)
            if layer_weight is None:
                pass
            elif c == nn.Conv2d or c == nn.Linear:
                if input_idx is None:
                    input_idx = range(layer_weight.shape[1])
                if output_idx is None:
                    output_idx = range(layer_weight.shape[0]) 
                layer_weight = layer_weight[output_idx][:,input_idx]
            elif c == nn.BatchNorm2d:
                if input_idx is None:
                    input_idx = range(layer_weight.shape[0])
                layer_weight = layer_weight[input_idx]
            else:
                print("Need weight dimension reduction for module: ", n)
                quit()

            if layer_weight is not None:
                print(n, " : ", i.shape, o.shape, layer_weight.shape)
            else:
                print(n, " : ", i.shape, o.shape)

            #torch.save(i, os.path.join(log_dir, f'{n}_input.pt'))
            #torch.save(o, os.path.join(log_dir, f'{n}_output.pt'))
            #torch.save(layer_weight, os.path.join(log_dir, f'{n}_weight.pth'))
            info[n] = {}
            info[n]["class"] = c.__name__
            info[n]["input"] = i.tolist()
            info[n]["output"] = o.tolist()
            info[n]["weight"] = layer_weight.tolist() if layer_weight is not None else None

        #torch.save(state_dict, os.path.join(log_dir, 'weights.pth'))
        with open(os.path.join(log_dir, model_name+'_info.json'), "w") as f:
            json.dump(info, f)

if __name__ == "__main__":
    for model in ['resnet18']:#, 'alexnet', 'googlenet', 'vgg16']:
        inference(f'./svelte-app/public/output/', './test_image/cat/image_1.jpg', model)

