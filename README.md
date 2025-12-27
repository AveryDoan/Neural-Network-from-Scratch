# Neural Network from Scratch
## Check it out: https://averydoan.github.io/Neural-Network-from-Scratch/

A modular and educational implementation of a neural network built only with `numpy`. This project is designed to help you understand the inner workings of neural networks by breaking them down into customizable components.

## Key Features

- **Modular Architecture**: Components like `Linear`, `ReLU`, and `Sequential` are built as independent blocks, similar to PyTorch.
- **Educational Comments**: Every layer and math operation is documented to explain *why* it works.
- **Interactive Visualizer**: A premium web dashboard to watch the network learn in real-time.
- **Gradient Checking**: Includes scripts to mathematically verify backpropagation accuracy.

## roject Structure

- `neural_network.py`: The core engine containing all modular layers and optimizers.
- `docs/`: A standalone web application for interactive training visualization (hosted on GitHub Pages).
- `main.ipynb`: A guided Jupyter notebook for experimentation and customization.
- `test_modules.py`: A automated test suite for math verification and convergence.

## The Data
This project uses **Synthetic Gaussian Blobs** by default.
- **Inputs**: 2D coordinates (easy for visualization).
- **Classes**: 3 distinct clusters.
- **Split**: 80% training / 20% testing.
*(Note: The Python notebook also includes code to easily switch to the MNIST handwritten digits dataset!)*

## How to Use

### 1. Interactive Web Dashboard (Live)
Watch the network in action instantly:
- **Live Version**: [https://averydoan.github.io/Neural-Network-from-Scratch/](https://averydoan.github.io/Neural-Network-from-Scratch/)
- **Local Version**: Open **`docs/index.html`** in your browser.
- Adjust parameters like Hidden Neurons and Learning Rate in the sidebar.
- Click **"Start Training"** to see the loss curve and connections update live.

### 2. Jupyter Notebook
For a deeper dive into the code:
- Open **`main.ipynb`**.
- Modify the `HIDDEN_DIM`, `LEARNING_RATE`, and `EPOCHS` in the customization cell.
- Run all cells to train and see visual predictions.

### 3. Running Math Verification
To ensure the implementation is correct:
```zsh
python3 test_modules.py
```

