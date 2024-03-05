import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import json
import os

import numpy as np
from PIL import Image


layer_inputs = []
layer_outputs = []
layer_names = []
layer_classes = []

check_list_class = [nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d, nn.AdaptiveAvgPool2d, nn.Linear]

# conv residual avgpool linear inception
module_start_name_dict = {
        "vgg16": [["features.0", "features.5", "features.10", "features.17", "features.24", 
                  "avgpool", "classifier.0"], 
                  ["conv","conv","conv","conv","conv","avgpool","linear"]],
        "alexnet": [["features.0", "features.3", "features.6", "avgpool", "classifier.1"],
                    ["conv","conv","conv","avgpool","linear"]],
        "resnet18": [["conv1", "layer1.0.conv1", "layer1.1.conv1", "layer2.0.conv1", "layer2.1.conv1", 
                   "layer3.0.conv1", "layer3.1.conv1", "layer4.0.conv1", "layer4.1.conv1", "avgpool", "fc"],
                   ["conv","residual","residual","residual","residual",
                       "residual","residual","residual","residual","avgpool","linear"]],
        "googlenet": [["conv1.conv", "conv2.conv", "inception3a.branch1.conv", "inception3b.branch1.conv", 
                      "inception4a.branch1.conv", "inception4b.branch1.conv", "inception4c.branch1.conv", 
                      "inception4d.branch1.conv", "inception4e.branch1.conv", "inception5a.branch1.conv",
                      "inception5b.branch1.conv", "avgpool", "fc"], 
                      ["conv","conv","inception","inception","inception","inception","inception","inception",
                      "inception","inception","inception","avgpool","linear"]]
        }

def insert_module_info(info, model_name):
    order_index = 0
    module_index = -1
    layer_index = 0
    for k in info.keys():
        if k in module_start_name_dict[model_name][0]:
            module_index += 1
            layer_index = 0 
        else:
            layer_index += 1
        info[k]['order_index'] = order_index
        info[k]['module_index'] = module_index
        info[k]['layer_index'] = layer_index
        info[k]['module_name'] = module_start_name_dict[model_name][1][module_index]
        order_index += 1

def hook_fn(m, i, o):
    if m._layer_name in layer_names:
        layer_names.append(m._layer_name + "2")
    else:
        layer_names.append(m._layer_name)
    layer_classes.append(m.__class__)
    layer_inputs.append(i[0].squeeze(0).detach().clone())
    layer_outputs.append(o[0].detach().clone())
    print(m._layer_name)
    #if "relu" in m._layer_name or "bn2" in m._layer_name:
    #    breakpoint()

def check_layer(name, module):
    if not hasattr(module, "_modules"):
        return True
    if isinstance(module, nn.Sequential):
        return False
    for c in check_list_class:
        if isinstance(module, c):
             return True
    return False

hook_handles = []
def hook_all_layers(net, name=""):
    split = "" if name == "" else "."
    for lname, layer in net._modules.items():
        if not check_layer(lname, layer):
            hook_all_layers(layer, name=f"{name}{split}{lname}")
        else:
            handle = layer.register_forward_hook(hook_fn)
            hook_handles.append(handle)
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
    ReLU_inplace_to_False(model)
    model.eval()

    hook_all_layers(model)
    
    return model

def get_layer_state(state_dict, layer_name):
    result = dict()
    for k in state_dict.keys():
        if k.startswith(layer_name):
            result[k] = state_dict[k]
    return result

def ReLU_inplace_to_False(module):
    # https://stackoverflow.com/questions/74124725/setting-relu-inplace-to-false
    #for layer in module._modules.values():
    #    if isinstance(layer, nn.ReLU):
    #        layer.inplace = False
    #    ReLU_inplace_to_False(layer)
    for layer in module._modules.values():
        if hasattr(layer, "inplace"):
            layer.inplace = False
        ReLU_inplace_to_False(layer)

def inference(log_dir, data_dir, model_name): 
    os.makedirs(log_dir, exist_ok=True)

    model = load_pretrained_model(model_name, 0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # need inplace=False for proper hook results
    #ReLU_inplace_to_False(model)
    model.to(device)
    #model.eval()

    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(227 if model_name == "alexnet" else 224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(data_dir)
    data = data_transforms(img)
    data = data.unsqueeze(0)
    info = {}

    with torch.no_grad():
        data = data.to(device)
        layer_inputs.clear()
        layer_outputs.clear()
        outputs = model(data)

        #for h in hook_handles:
        #    h.remove()

        state_dict = model.state_dict()
        prev_input_idx = None
        prev_output_idx = None
        prev_bn_output = None
        residual_module_input = None

        for n in layer_names:
            info[n] = {}
        insert_module_info(info, model_name)

        # create info
        for cnt, (i, o, n, c) in enumerate(zip(layer_inputs, layer_outputs, layer_names, layer_classes)):
            i = i.cpu().numpy()
            o = o.cpu().numpy()
            info[n]["input"] = i
            info[n]["output"] = o
            info[n]["class"] = c
           
            layer_state = get_layer_state(state_dict, n)
            layer_weight = None
            if n+".weight" in layer_state:
                layer_weight = layer_state[n+".weight"].cpu().numpy()
                print(n, " : ", c, i.shape, o.shape, layer_weight.shape, o.max(), o.min())
            else:
                print(n, " : ", c, i.shape, o.shape, o.max(), o.min())    
            info[n]["weight"] = layer_weight if layer_weight is not None else None

        # get residual info
        for n in info:
            # count residual module size
            if info[n]["module_name"] == "residual":
                midx = info[n]["module_index"]
                lidx = info[n]["layer_index"]
                msize = sum([info[k]["module_index"] == midx for k in info])
                # if last layer set first layer input or downsample output as identity
                ds_layer_name = n[:-6]+".downsample.1"
                if lidx + 1 == msize:
                    ds_layer_name = n[:-6]+".downsample.1"
                    if ds_layer_name in info:
                        identity = [info[ds_layer_name]["output"]]
                    else:
                        identity = [info[k]["input"] for k in info if info[k]["module_index"] == midx and info[k]["layer_index"] == 0]
                    info[n]["identity"] = identity[0]
                    print(n, info[n]["module_index"], identity[0].shape)

        # set sampling index
        for n in info:
            input_idx = None
            output_idx = None
            softmax_output = None
            
            i = info[n]["input"]
            o = info[n]["output"]
            c = info[n]["class"]
            
            if c == nn.Linear:
                # Linear: first 1024
                # Last Linear: top 10 cls
                if i.shape[0] > 1024:
                    input_idx = range(1024)
                if n == layer_names[-1]:
                    output_idx = (-o).argsort()[:10]
                    y = np.exp(o - np.max(o))
                    softmax_output = y / np.sum(np.exp(o)) * np.exp(np.max(o))
                    softmax_output = softmax_output[output_idx]
                elif o.shape[0] > 1024:
                    output_idx = range(1024)
            elif c in [nn.ReLU, nn.BatchNorm2d, nn.MaxPool2d, nn.AdaptiveAvgPool2d]:
                # get previous output index
                k = list(info)[info[n]['order_index'] - 1]
                input_idx = info[k]['output_index']
                output_idx = info[k]['output_index']
            else: # Conv
                # first 8 or get previous output index 
                if i.shape[0] > 8:
                    if prev_output_idx is not None:
                        input_idx = prev_output_idx
                    else:
                        input_idx = range(8)
                # set based on sampling method 
                if o.shape[0] > 8:
                    # get activation value
                    output_idx = range(8)
                    for ii in range(info[n]['order_index'], len(info)):
                        if info[list(info)[ii]]["class"] == nn.ReLU:
                            activation = info[list(info)[ii]]["output"]
                            avg_activation = np.mean(activation, axis=(1,2))
                            output_idx = (-avg_activation).argsort()[:8]
                            break

            layer_weight = info[n]['weight']
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
                print("Need weight dimension reduction for layer: ", n)
                quit()
            
            input_idx = np.array(input_idx)
            output_idx = np.array(output_idx)
            i = i[input_idx]
            o = o[output_idx]
            print(input_idx)
            print(output_idx)
            info[n]['class'] = info[n]['class'].__name__
            info[n]['input'] = i.tolist()
            info[n]['output'] = o.tolist()
            info[n]['input_index'] = input_idx.tolist()
            info[n]['output_index'] = output_idx.tolist()
            if n == layer_names[-1]:
                info[n]['softmax_output'] = softmax_output.tolist()
        
            if layer_weight is not None:
                info[n]['weight'] = layer_weight.tolist()
                print(n, " : ", i.shape, o.shape, layer_weight.shape)
            else:
                print(n, " : ", i.shape, o.shape)
            #print(input_idx)
            #print(output_idx)

        # set residual info / apply sampling index
        for cnt, n in enumerate(info):
            if "identity" in info[n]:
                prev_output_index = info[list(info)[cnt-1]]["output_index"]
                info[n]["identity"] = info[n]["identity"][prev_output_index].tolist()
        
        with open(os.path.join(log_dir, model_name+'_info.json'), "w") as f:
            json.dump(info, f)
    with open(os.path.join(log_dir, 'module_info.json'), "w") as f:
        json.dump(module_start_name_dict, f)

def get_imagenet_data():
    images = []
    path = "./imagenet-sample-images"
    for fname in sorted(os.listdir(path)):
        if fname.endswith(".JPEG"):
            images.append(os.path.join(path, fname))
    return images

if __name__ == "__main__":
    test_image = "./test_image/cat/image_1.jpg"

    imagenet_data = get_imagenet_data()[:5]
    for model in ['alexnet', 'resnet18', 'googlenet', 'vgg16']:
        for index, data in enumerate(imagenet_data):
            #label = IMAGENET_CLASSES[index]
            inference(f"./svelte-app/public/output/{index}/", data, model)
            layer_inputs = []
            layer_outputs = []
            layer_names = []
            layer_classes = []
