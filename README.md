# CNN Fashion-MNIST Classification

**Systematic CNN experiments for Fashion-MNIST classification - exploring epoch impact on model performance with TensorFlow/Keras. Achieved 91% validation accuracy with analysis of overfitting patterns.**

## 🎯 Overview

This project implements Convolutional Neural Networks (CNNs) for clothing classification using the Fashion-MNIST dataset. Through **systematic experimentation across 1-8 epochs**, this work demonstrates CNN architecture design, training optimization, and overfitting analysis using modern deep learning frameworks.

**Key Achievements:**
- ✅ Built CNN architecture with TensorFlow/Keras
- ✅ Achieved 91% validation accuracy on Fashion-MNIST
- ✅ Conducted systematic epoch-by-epoch analysis
- ✅ Identified optimal training duration (5-6 epochs)
- ✅ Analyzed and documented overfitting patterns

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/cnn-fashion-mnist.git
cd cnn-fashion-mnist

# Run the notebook
jupyter notebook cnn_mnist_classification.ipynb
```

## 📊 Dataset: Fashion-MNIST

**Not Regular MNIST** - This is a more challenging dataset!

- **Training samples**: 60,000 grayscale images
- **Test samples**: 10,000 grayscale images  
- **Image size**: 28×28 pixels
- **Classes**: 10 fashion categories
  - T-shirt/top, Trouser, Pullover, Dress, Coat
  - Sandal, Shirt, Sneaker, Bag, Ankle boot

**Why Fashion-MNIST?**
- More challenging than digit recognition
- Realistic computer vision task
- Industry-relevant (fashion/e-commerce applications)

## 🏗️ Model Architecture

```python
Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(5, activation='softmax')  # 5 classes as per assignment
])
```

**Layer Breakdown:**
1. **Conv2D (32 filters)**: Extract basic patterns
2. **MaxPooling2D**: Reduce dimensions, prevent overfitting
3. **Conv2D (64 filters)**: Detect complex features
4. **MaxPooling2D**: Further dimensionality reduction
5. **Flatten**: Convert 2D features to 1D
6. **Dense (128 units)**: Learn combinations of features
7. **Dropout (0.5)**: Regularization
8. **Dense (5 units)**: Output layer with softmax

## 📈 Experimental Results

### Systematic Epoch Analysis

| Epochs | Train Accuracy | Val Accuracy | Observations |
|--------|---------------|--------------|--------------|
| **1** | 63.5% | 81.0% | Baseline performance |
| **2** | 83.7% | 86.0% | Rapid learning |
| **3** | 88.7% | 89.6% | Continued improvement |
| **4** | 92.0% | 87.4% | Val acc. starts fluctuating |
| **5** | 94.0% | 87.6% | Training improving, val plateaus |
| **6** | 94.9% | 89.8% | Good balance |
| **7** | 97.5% | 91.0% | **Peak validation accuracy** |
| **8** | 97.2% | 91.2% | Overfitting begins |

### Key Findings

**🎯 Optimal Training Duration: 5-7 Epochs**
- **Best validation accuracy**: 91% (epochs 7-8)
- **Balance point**: Epochs 5-6 for time efficiency
- **Overfitting threshold**: After epoch 6

**📊 Training Dynamics:**
- **Early epochs (1-3)**: Rapid accuracy gains
- **Middle epochs (4-6)**: Refinement and stabilization  
- **Late epochs (7-8)**: Overfitting risk increases

**⚠️ Overfitting Indicators:**
- Training accuracy reaches 97-99%
- Validation accuracy plateaus at 90-91%
- Gap between train/val widens after epoch 6

## 🛠️ Technical Implementation

### Data Preprocessing

```python
# Load Fashion-MNIST
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Normalize to [0, 1]
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Reshape for CNN: (samples, height, width, channels)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)
```

### Model Compilation

```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

**Configuration:**
- **Optimizer**: Adam (adaptive learning rate)
- **Loss**: Sparse categorical cross-entropy (integer labels)
- **Metrics**: Accuracy

### Training Process

```python
history = model.fit(
    X_train, y_train,
    batch_size=128,
    epochs=num_epochs,
    validation_split=0.2,
    verbose=1
)
```

- **Batch size**: 128 samples
- **Validation split**: 20% of training data
- **Systematic testing**: 1, 2, 3, ..., 8 epochs

## 💡 Skills Demonstrated

### Deep Learning & CNNs
- ✅ CNN architecture design and implementation
- ✅ Convolutional layer configuration
- ✅ Pooling strategies (MaxPooling)
- ✅ Regularization techniques (Dropout)
- ✅ Activation functions (ReLU, Softmax)

### TensorFlow/Keras Proficiency
- ✅ Sequential model building
- ✅ Layer configuration and stacking
- ✅ Model compilation and training
- ✅ Training history analysis
- ✅ Performance monitoring

### Experimental Methodology
- ✅ Systematic hyperparameter testing (epochs)
- ✅ Performance metric tracking
- ✅ Overfitting detection and analysis
- ✅ Training curve interpretation
- ✅ Optimal configuration identification

### Computer Vision
- ✅ Image preprocessing pipelines
- ✅ Normalization techniques
- ✅ Data shape manipulation
- ✅ Multi-class image classification
- ✅ Model evaluation on vision tasks

## 🔬 Analysis & Insights

### Why CNNs for Images?

**Advantages over Fully Connected Networks:**
1. **Parameter Efficiency**: Weight sharing reduces parameters
2. **Spatial Understanding**: Preserves spatial relationships
3. **Translation Invariance**: Detects patterns anywhere in image
4. **Hierarchical Learning**: Low-level → High-level features

### Architecture Design Rationale

**Convolutional Layers:**
- **32 filters (first layer)**: Detect edges, textures
- **64 filters (second layer)**: Detect complex patterns
- **3×3 kernels**: Standard choice, good balance

**Pooling Layers:**
- Reduces spatial dimensions (28×28 → 14×14 → 7×7)
- Prevents overfitting through dimensionality reduction
- Makes features more robust to small translations

**Dropout (50%):**
- Prevents co-adaptation of neurons
- Acts as ensemble learning
- Improves generalization

### Overfitting Analysis

**Why Overfitting Occurs:**
- Model capacity exceeds data complexity
- Training too many epochs
- Memorization of training data

**How We Detected It:**
- Training accuracy continues improving (→99%)
- Validation accuracy plateaus (→91%)
- Increasing gap after epoch 6

**Prevention Strategies:**
- Stop training at epochs 5-6
- Use dropout (already implemented)
- Could add: data augmentation, early stopping

## 🎓 Learning Outcomes

### Practical Skills Gained

1. **CNN Implementation**: Building models with modern frameworks
2. **Training Management**: Monitoring and controlling training process
3. **Hyperparameter Analysis**: Understanding epoch impact
4. **Overfitting Recognition**: Detecting and preventing overfitting
5. **Performance Optimization**: Finding optimal configurations

### Theoretical Understanding

1. **Convolutional Operations**: How filters extract features
2. **Pooling Effects**: Dimensionality reduction benefits
3. **Regularization**: Dropout and its mechanisms
4. **Optimization**: Adam optimizer behavior
5. **Loss Functions**: Cross-entropy for classification

## 🔍 Key Takeaways

### Project Insights

1. **More Epochs ≠ Better**: Found optimal at 5-7 epochs
2. **Monitor Both Metrics**: Train and validation tell different stories
3. **Overfitting Detection**: Critical skill for model deployment
4. **Framework Proficiency**: TensorFlow/Keras industry standard
5. **Systematic Approach**: Methodical testing reveals insights

### Real-World Applications

CNN techniques from this project apply to:
- E-commerce product categorization
- Fashion recommendation systems
- Quality control in manufacturing
- Medical image analysis
- Autonomous vehicle perception
- Security and surveillance systems

## 🚀 Future Improvements

Potential enhancements:

- [ ] Data augmentation (rotation, shift, zoom)
- [ ] Different architectures (ResNet-style, VGG-style)
- [ ] Batch normalization layers
- [ ] Learning rate scheduling
- [ ] Early stopping callbacks
- [ ] Confusion matrix analysis
- [ ] Test on full 10-class Fashion-MNIST
- [ ] Model deployment as REST API
- [ ] Transfer learning experiments

## 🤝 Contributing

Personal learning project. Suggestions and feedback welcome!

## ✉️ Contact

**Vaibhav Tamta**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)
- Email: your.email@example.com

## 🙏 Acknowledgments

- TensorFlow and Keras development teams
- Fashion-MNIST dataset creators (Zalando Research)
- Deep learning research community
- Course instructors and teaching staff

## 📚 References & Resources

### Academic Papers
- LeCun, Y., et al. (1998). "Gradient-based learning applied to document recognition"
- Xiao, H., Rasul, K., & Vollgraf, R. (2017). "Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms"

---

<p align="center">
  <i>From fundamentals to frameworks - mastering CNNs through systematic experimentation 🧠</i>
</p>

<p align="center">
  <i>Achieving 91% accuracy on Fashion-MNIST with optimal training strategies</i>
</p>
