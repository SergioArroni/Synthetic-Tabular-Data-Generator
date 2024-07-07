# Synthetic Tabular Data Generator Based on Diffusion Models

## Description

This repository contains the code and data used for a research project by Sergio Arroni. The goal of the project is to create a synthetic tabular data generator using diffusion models. Different datasets can be found in `./all_results/synthetic` within their respective folders.

## Repository Structure

- `data/`: Contains the data used in the project, divided into subfolders by type.
- `img/`: Images used for training and analysis.
- `prep/`: Data preprocessing scripts.
- `all_results/`: Results obtained during project development.
- `final_results_tmp/`: Temporary results of the model.
- `doc/`: Research work and resources used.
- `main.py`: Main script of the project.
- `requirements.txt`: List of dependencies needed to run the project.
- `.gitignore`: Files and folders ignored by git.

## Installation

### Cloning the Repository and Installing Dependencies

1. **Clone the repository**:

   ```bash
   git clone https://github.com/SergioArroni/TFM.git
   cd TFM
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

### Setting Up PyTorch and CUDA 12

3. **Create and activate a virtual environment**:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

4. **Install PyTorch with CUDA 12 support**:
   Follow the instructions on the [PyTorch website](https://pytorch.org/get-started/locally/):

   ```bash
   pip install torch torchvision torchaudio
   ```

5. **Verify CUDA installation**:
   Follow the instructions on the [CUDA Toolkit website](https://developer.nvidia.com/cuda-toolkit):
   ```bash
   nvcc --version
   ```

## Usage

1. **Enter the virtual environment**:

   ```bash
   .\.venv\Scripts\activate
   ```

2. **Set the PYTHONPATH** (in case of path issues):

   ```bash
   $env:PYTHONPATH="Your/Code/Path"
   ```

3. **Run the preprocessing script**:

   ```bash
   python prep/prep.py
   ```

4. **Modify `main.py` as needed for the experiment and execute**:
   ```bash
   python main.py
   ```

## Explanation of Main Components

### `main.py`

The main script initializes seeds, preprocesses data if required, loads data, sets up and trains a diffusion model, and evaluates the model.

- **Boolean Flags**: `prep` control data preprocessing, `load` controls whether to load a pre-trained model or train a new one.
- **Training and Evaluation**: Uses PyTorch for training with options to adjust model parameters like epochs and batch size.

### `test.py`

The `Test` class in `test.py` evaluates the model's performance in terms of efficiency, quality, and privacy.

- **Evaluation Methods**: Includes methods for generating synthetic data, evaluating efficiency, quality, and privacy.
- **Customization**: The tests can be modified depending on the specific evaluation criteria needed for the experiment.

### `diffusion_model.py`

The `DiffusionModel` class defines the architecture and training procedure for the diffusion model.

- **Components**:
  - **Encoder**: Converts input into a higher-dimensional latent representation.
  - **Transformer**: Processes the latent representation.
  - **Decoder**: Converts the processed latent representation back into the original input size.
- **Diffusion Process**: Iteratively adds noise controlled by beta values and adjusted based on the standard deviation of the activations.
- **Training**: Uses PyTorch for backpropagation and gradient clipping.

### Extendability

The project is structured to easily allow the addition of new models. This is achieved by using a strategy pattern, which ensures that new models can be integrated without altering the existing codebase significantly. This makes the project flexible and adaptable to various experimental needs.

## Contributions

To contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/new-feature`).
3. Make your changes and commit them (`git commit -am 'Add new feature'`).
4. Push your changes (`git push origin feature/new-feature`).
5. Open a Pull Request.

## Contact

Sergio Arroni - [sergioArroni@outlook.com](sergioArroni@outlook.com)

## Resources

- More details can be found in the [official repository](https://github.com/SergioArroni/TFM).
