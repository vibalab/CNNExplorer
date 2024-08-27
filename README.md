# Vis4CNN

This repository provides tools to load and inference Convolutional Neural Network (CNN) models using PyTorch and Python, as well as visualize the intermediate tensors using Svelte and JavaScript.

## Features

1. **CNN Model Inference**: Load pre-trained CNN models and perform inference on input data.
2. **Intermediate Tensor Visualization**: Visualize intermediate tensors generated during model inference using an intuitive web interface.

## Requirements

- Python 3.11.5
- PyTorch 2.0.1
- Svelte
- JavaScript

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Negota/vis4cnn.git
   ```

2. Install conda environment and python dependencies:

   ```bash
   conda create -n vis4cnn python==3.11.5
   pip install -r requirements.txt
   ```
3. Download imagenet sample images

   ```bash
   cd server
   git clone https://github.com/EliSchwartz/imagenet-sample-images
   cd ..
   ```

4. Install nvm and NodeJS following [link](https://github.com/nvm-sh/nvm?tab=readme-ov-file#installing-and-updating)

   ```bash
   curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
   source ~/.bashrc
   nvm install node
   ```

5. Install Svelte and JavaScript dependencies:

   ```bash
   cd client
   npm install
   cd ..
   ```


## Usage

### CNN Model Inference

1. Download pre-trained CNN model weights.
2. Use the provided Python scripts to perform inference on your desired input data.
3. Run flask server app

Example:

```bash
cd server
mkdir weights
python get_pretrained.py # resnet18, 0
python inference/main.py # fix main function to specifiy imagenet samples and models
python app.py
```

### Intermediate Tensor Visualization

1. Start the Svelte development server:

   ```bash
   cd client
   npm run dev
   ```

2. Open your web browser and navigate to `http://localhost:8080`.

## Contributing

Contributions are welcome! Please feel free to open issues or pull requests for any improvements or features you'd like to see.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
