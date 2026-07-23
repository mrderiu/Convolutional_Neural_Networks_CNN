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
