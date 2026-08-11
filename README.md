### Convolutional Neural Networks CNNs
After understanding Multilayer Perceptrons (MLPs), the next natural step in Deep Learning is learning Convolutional Neural Networks (CNNs). MLPs work well with tabular data, where each input feature is independent. However, they are not well suited for image data because they ignore the spatial relationships between neighboring pixels. Images contain a large amount of local structure. Nearby pixels are often related and form meaningful patterns such as edges, corners, textures, and shapes. Convolutional Neural Networks were specifically designed to exploit these spatial relationships.

Today, CNNs are one of the most important architectures in Computer Vision and are widely used for image classification, object detection, image segmentation, facial recognition, medical imaging, and many other applications.

Consider a color image of size 128 × 128 pixels.

An MLP would first flatten the image into a single vector:
```
128 × 128 × 3 --> 49152 values
```
The first fully connected layer would then connect every input pixel to every neuron.

For example,

nn.Linear(49152, 256)

requires

49152 × 256 = 12,582,912

trainable weights in only one layer.

This approach has several disadvantages:

* A very large number of parameters.
* High memory consumption.
* Slow training.
* Increased risk of overfitting.
* No understanding of the spatial arrangement of pixels.

An MLP treats every pixel as an independent value. It has no notion that two neighboring pixels are related or that together they may form an edge or part of an object. Instead of connecting every pixel to every neuron, CNNs process images using small local filters. These filters move across the image and learn to recognize visual patterns. Rather than learning millions of independent connections, a CNN learns a relatively small number of reusable filters. These filters can detect features such as:

* Horizontal edges
* Vertical edges
* Corners
* Curves
* Textures
* Simple shapes

As the network becomes deeper, these simple features are combined into increasingly complex representations.

For example:
```
Pixels --> Edges --> Corners  --> Textures --> Eyes --> Faces --> Entire Objects
```
This hierarchical learning process is one of the key ideas behind Convolutional Neural Networks.

#### Building a Convolutional Neural Network in PyTorch
Consider the following convolutional neural network:

```
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
```
This network is divided into two parts:
* Feature Extractor (self.features): Learns visual patterns from the image.
* Classifier (self.classifier): Uses the extracted features to predict the final class.

* **Conv2d**
nn.Conv2d implements a two-dimensional convolutional layer. Instead of connecting every pixel to every neuron as in an MLP, a convolution layer applies a small learnable filter (kernel) that slides across the image and detects visual patterns.

The syntax is:
```
nn.Conv2d(
    in_channels, # Number of input channels.  
    out_channels, # Number of filters (feature maps) produced.
    kernel_size, # Size of the convolution filter (e.g., 3×3 or 5×5).
    stride=1, # Number of pixels the filter moves at each step.
    padding=0 # Number of pixels added around the image border.
)
```
<<<<<<< HEAD
=======
a more modern implementation is
```
model = nn.Sequential(
    nn.Linear(8, 12),
    nn.ReLU(),
    nn.Linear(12, 8),
    nn.ReLU(),
    nn.Linear(8, 1)
)

loss_fn = nn.BCEWithLogitsLoss()
```
BCEWithLogitsLoss combines the Sigmoid activation and Binary Cross Entropy into a single operation that is more numerically stable and is the recommended approach in PyTorch.

#### More Features

* **Dropout**
As neural networks become larger, they may memorize the training data instead of learning general patterns. Dropout is a regularization technique that randomly disables a percentage of neurons during training.

Example:
```
nn.Linear(32, 16),
nn.ReLU(),
nn.Dropout(0.2)
```
This forces the network to learn more robust representations and reduces overfitting.

* **Normalize Activations**
BatchNorm1d normalizes the outputs of a layer during training. A common pattern is:
```
Linear --> BatchNorm --> ReLU --> Dropout
```
Batch Normalization often stabilizes training and allows the optimizer to converge faster.

* **Dataset and DataLoader**
Instead of manually creating mini-batches with array slicing, PyTorch provides the Dataset and DataLoader classes.

DataLoader automatically:

* creates mini-batches,
* shuffles the training data,
* loads batches efficiently,
* simplifies the training loop.

These classes are the standard way of feeding data into neural networks.

* **Imbalanced Data**
Many real-world datasets contain significantly more samples from one class than the other. For binary classification problems, PyTorch provides the pos_weight parameter in BCEWithLogitsLoss to give more importance to the minority class during training. This is particularly useful for applications such as fraud detection, anomaly detection, and medical diagnosis.

* **Early Stopping**
Training for too many epochs may cause the model to overfit the training data. Early stopping monitors the validation loss and automatically stops training when the model no longer improves. It also saves the best-performing version of the model.


### Project Structure

```
main.py                  # Orchestrates the complete pipeline: trains, compares,
                         # and saves the best model for production
src/
├── config.py            # Paths, columns, hyperparameters, and model selection
├── data_loader.py       # CSV loading and train/validation/test split
├── preprocessing.py     # Data cleaning and ColumnTransformer
│                        # (imputation + scaling + one-hot encoding)
├── models.py            # MLP architectures + build_model() factory
├── early_stopping.py    # Early Stopping implementation
├── train.py             # Training loop (forward, loss, backward, optimizer)
├── evaluate.py          # Validation and test evaluation metrics
└── experiment_log.py    # Logs each experiment run to logs/experiments.csv
predict.py               # Production inference using the saved winning model
```

### Model Variants (`src/models.py`)

Both architectures return **logits** (without applying `Sigmoid` to the output), because `train.py` always uses `nn.BCEWithLogitsLoss`. This loss function already incorporates the sigmoid operation in a numerically stable way and also allows the minority class to be weighted using `pos_weight`.

Combining a model that already applies `Sigmoid` with `BCEWithLogitsLoss` would apply the sigmoid function twice and produce incorrect output probabilities. For this reason, the project consistently follows the logits-only approach across all model variants.

| Variant | Architecture | When to Use |
| --- | --- | --- |
| `"simple"` | `Linear → ReLU` ×3 | Baseline model without regularization. Suitable as a starting point or reference model. |
| `"regularized"` | `Linear → BatchNorm → ReLU → Dropout` ×2 | Adds normalization and regularization to reduce overfitting. Recommended as the default architecture. |

The active variant can be configured through `config.MODEL_VARIANT` and is instantiated using the `build_model(name, input_dim)` factory, without requiring any changes to `main.py`.

### Architecture Comparison and Best Model Selection

`main.py` does not train a single architecture. Instead, it iterates through **all** model variants registered in `MODEL_REGISTRY` (`src/models.py`), trains each architecture independently, and compares their performance.

This design makes the project easily extensible: adding a new architecture to `MODEL_REGISTRY` automatically includes it in the comparison process without requiring any changes to `main.py`.

The winning model is selected based on its **validation F1 score**, rather than accuracy or test performance:

- **F1 instead of accuracy** because the positive class (`income > 50K`) is underrepresented in the dataset. Accuracy can therefore be misleading when evaluating an imbalanced classification problem.
- **Validation instead of test** because the test set is reserved exclusively for the final evaluation of the selected model. Using test performance for model selection would introduce information leakage from the test set into the model selection process.

### Experiment Tracking (`src/experiment_log.py`)

Each trained architecture adds a new row to:

`logs/experiments.csv`

The log contains information such as:

- Timestamp
- Model variant
- Validation accuracy
- Validation precision
- Validation recall
- Validation F1
- Test accuracy
- Test precision
- Test recall
- Test F1

The file is created automatically if it does not already exist.

Each execution of `main.py` appends new rows without overwriting previous results, providing a historical record of all experiment runs.

### Production Artifacts and `predict.py`

Once all architectures have been compared, `main.py` saves everything required to perform inference with the winning model under:

`models/production/`

The following artifacts are generated:

- `model.pt` — trained model weights (`state_dict`).
- `preprocessor.joblib` — the `ColumnTransformer` fitted on the training dataset.
- `metadata.json` — metadata describing the winning architecture and the `input_dim` used to construct it. This information is required to reconstruct the neural network before loading its weights.

`predict.py` loads these production artifacts and exposes a `predict(df)` function that can generate predictions for new, unseen data without retraining the model.

```bash
python predict.py
>>>>>>> beea004 (updated the READ.md)


