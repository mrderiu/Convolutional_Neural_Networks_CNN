### Convolutional Neural Networks CNNs
After understanding Multilayer Perceptrons (MLPs), the next natural step in Deep Learning is learning Convolutional Neural Networks (CNNs).

MLPs work well with tabular data, where each input feature is independent. However, they are not well suited for image data because they ignore the spatial relationships between neighboring pixels.

Images contain a large amount of local structure. Nearby pixels are often related and form meaningful patterns such as edges, corners, textures, and shapes. Convolutional Neural Networks were specifically designed to exploit these spatial relationships.

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

A very large number of parameters.
High memory consumption.
Slow training.
Increased risk of overfitting.
No understanding of the spatial arrangement of pixels.

An MLP treats every pixel as an independent value. It has no notion that two neighboring pixels are related or that together they may form an edge or part of an object.
