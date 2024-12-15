# Landslide Detection Using Deep Learning

This repository contains code for training and evaluating a landslide detection model using deep learning techniques. The primary goal of this project is to classify satellite images into landslide and non-landslide regions, leveraging a U-Net model architecture.

---

## Features
- **Dataset Handling**: Load and preprocess labeled and unlabeled landslide datasets.
- **Model Architecture**: Utilize a U-Net-based neural network for segmentation tasks.
- **Training**: Train the model with support for multi-class classification.
- **Evaluation**: Compute key metrics such as precision, recall, F1-score, and overall accuracy.
- **Customization**: Modular design to easily integrate other models or datasets.

---

## Requirements

### Software Dependencies
The codebase is developed in Python and requires the following libraries:

- Python >= 3.8
- PyTorch >= 1.10
- NumPy
- h5py
- CUDA (if GPU support is required)

### Install Required Libraries
To install the required dependencies, use the following command:

```bash
pip install -r requirements.txt
```

---

## Dataset Preparation

The dataset should be in `.h5` format, with image-label pairs for training. Update the dataset path in the training arguments (`--data_dir`).

**Directory Structure Example**:
```
/scratch/Land4Sense_Competition_h5/
    ├── train.txt
    ├── test.txt
    ├── img1.h5
    ├── img2.h5
    └── ...
```

---

## Training the Model

To train the model, run the `train.py` script:

```bash
python train.py \
    --data_dir /path/to/dataset \
    --model_module model.Networks \
    --model_name unet \
    --train_list /path/to/train_list.txt \
    --test_list /path/to/test_list.txt \
    --input_size 128,128 \
    --num_classes 2 \
    --batch_size 32 \
    --num_steps 5000 \
    --learning_rate 0.001 \
    --snapshot_dir ./models
```

### Key Arguments
- `--data_dir`: Path to the dataset directory.
- `--model_module`: Module containing the model.
- `--model_name`: Name of the model.
- `--train_list`: File containing paths to training data.
- `--test_list`: File containing paths to test data.
- `--input_size`: Input image size (width, height).
- `--num_classes`: Number of output classes.
- `--batch_size`: Batch size for training.
- `--num_steps`: Total number of training steps.
- `--learning_rate`: Learning rate for the optimizer.
- `--snapshot_dir`: Directory to save model checkpoints.

---

## Evaluating the Model

Evaluation occurs every 500 steps during training. Key metrics are logged, including:
- **Precision**
- **Recall**
- **F1-score**
- **Overall Accuracy**

Checkpoints of the best-performing models are saved in the snapshot directory.

---

## Customization

### Model Integration
You can integrate other models by modifying the `--model_module` and `--model_name` arguments to point to the desired model's Python module and class.

### Dataset Adaptation
Modify the `LandslideDataSet` class in `dataset/landslide_dataset.py` to handle custom dataset formats.

---

## Contact

For questions or feedback, feel free to reach out via email or open an issue in this repository.

