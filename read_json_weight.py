import json
import numpy as np
import matplotlib.pyplot as plt

# Load the JSON file
with open('weights/resnet18_0_weights.json', 'r') as file:
    model_weights = json.load(file)

# Print all keys
print("Keys in the JSON file:")
for key in model_weights.keys():
    print(key)

# Access a specific element using a key
# Replace 'specific_key' with the actual key you want to access
specific_key = 'layer1.0.conv1.weight'  # Example key, adjust as needed
if specific_key in model_weights:
    specific_element = model_weights[specific_key]
    print(f"\nValue for '{specific_key}': {specific_element}")
    # For convolutional layers, you might need to select a specific filter and channel
    # Example: selecting the first filter and first channel
    weights = np.array(model_weights[specific_key])[0, 0, :, :]
else:
    print(f"\nKey '{specific_key}' not found in the JSON file.")
    
    # Plotting the 2D matrix
plt.imshow(weights, cmap='viridis')
plt.colorbar()
plt.title(f"Weights of {specific_key}")
plt.show()
plt.savefig('weight.png')
