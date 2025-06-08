# Handwritten Digit Recognition with NumPy: Building a Deep Neural Network from Scratch

![MNIST Digits](https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png)

## Overview
This project implements a **3-layer Deep Neural Network (DNN)** using only NumPy to classify handwritten digits from the MNIST dataset. By building all components from scratch—including forward/backward propagation, activation functions, and gradient descent—we demonstrate the foundational mechanics of deep learning without high-level frameworks like TensorFlow or PyTorch.

**Key Features**:
- Pure NumPy implementation (no DL frameworks)
- He initialization for efficient learning
- Mini-batch gradient descent optimization
- ReLU activation in hidden layers
- Softmax output layer with cross-entropy loss
- Achieves **95% accuracy** on validation set

## Project Structure
```
.
├── DNN_using_NumPy.ipynb                              # Main Jupyter notebook with implementation
├── Neural Networks and Deep Learning (Notes).pdf      # Notes explaining Mathematical part of Neural Networks
├── train.csv                                          # MNIST training data (CSV format)
├── README.md                                          # This documentation
└── requirements.txt                                   # Python dependencies
```

## How to Run
1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Requirements: NumPy, pandas, scikit-learn

2. **Download Data**:
   - Place `train.csv` in the project directory (available on [Kaggle](https://www.kaggle.com/competitions/digit-recognizer/data))

3. **Execute the Notebook**:
   ```bash
   jupyter notebook DNN_using_NumPy.ipynb
   ```

4. **Training Process**:
   - 3-layer network (128 → 64 → 10 units)
   - 20 epochs with batch size 64
   - Learning rate: 0.01
   - Loss decreases from 1.12 to 0.13

## Results
| Epoch | Loss     | Validation Accuracy |
|-------|----------|---------------------|
| 1     | 1.1287   | -                   |
| 10    | 0.2068   | -                   |
| **20**| **0.1366**| **95.07%**          |

**Confusion Matrix** (Sample predictions vs actual labels):
```
        0    1    2    3    4    5    6    7    8    9
    0  813    0    3    1    0    1    6    1    3    0
    1    0  938    2    1    0    0    1    1    1    0
    2    1    2  822    5    1    0    1    4    1    1
    ... [full matrix in notebook]
```

## Key Components
### 1. Network Architecture
```python
Input (784) → Hidden 1 (128, ReLU) → Hidden 2 (64, ReLU) → Output (10, Softmax)
```

### 2. Core Functions
- **Initialization**: He weight initialization (`init_params()`)
- **Forward Propagation**: `forward_prop()`
- **Activation Functions**: ReLU and Softmax
- **Loss Calculation**: Cross-entropy (`cross_entropy()`)
- **Backpropagation**: `backward_prop()`
- **Parameter Update**: Gradient descent (`update_params()`)

### 3. Training
Mini-batch gradient descent with shuffling:
```python
for epoch in range(epochs):
    perm = np.random.permutation(n)
    for i in range(0, n, batch_size):
        # Forward pass, loss calculation, backprop, update
```

## Future Improvements
1. **Hyperparameter Tuning**:
   - Implement learning rate scheduling
   - Experiment with Adam optimizer
   
2. **Regularization Techniques**:
   - Add L2 regularization
   - Implement dropout layers

3. **Architecture Enhancements**:
   - Add batch normalization
   - Extend to convolutional layers (CNN)
   
4. **Deployment**:
   - Create web interface for real-time predictions
   - Optimize for mobile devices using ONNX

## Dependencies
- Python 3.7+
- NumPy 1.21+
- pandas 1.3+
- scikit-learn 1.0+

## Contributor
**Utkarsh Bhardwaj**  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Utkarsh284-blue)](https://www.linkedin.com/in/utkarsh284/)
[![GitHub](https://img.shields.io/badge/GitHub-utkarsh--284-lightgrey)](https://github.com/utkarsh-284)  
**Contact**: ubhardwaj284@gmail.com  
**Publish Date**: 8th June, 2025  

---

**Why This Project Stands Out**: By implementing every neural network component from scratch, we demystify deep learning fundamentals while achieving competitive performance. This serves as both an educational resource and a foundation for more complex architectures.
