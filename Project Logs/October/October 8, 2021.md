# Overview of today: 
## What I did:  
- Created model architecture. Basic summary of the model's layers:
    - 3 layers of 5x5 convolutional filters, each separated by a 2x2 max pooling layer.
    - A dropout layer (0.3) to dilute the weights
    - A flattening layer, to input into a fully connected layer with 512 possible outputs.
    - Finally, another 0.3 dropout layer to thin the weights more.
    - An output layer, consisting of an array of 4x62 arrays. Uses softmax activation (want the highest probability).
- Read some literature (Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks)
## What I now know:
- Flattening converts the inputs to a one-dimensional vector, and its output is often used as an input for a fully connected layer.
- Dropout is also referred to as "dilution", as the process of dropping or setting a certain amount of weights to 0 forces the remaining nodes to take more responsibility in activating upon detection of a certain feature.
    - It is often recommended to use dropout in large networks, as large networks may overfit if there are too few samples.
- GANs with certain elements of CNN architecture have the potential to be strong unsupervised learning candidates.
    - A form of ReLU called "leaky ReLu" is often used in GANs, as the discriminator improves too fast with normal ReLU, causing nonsensical outputs with high generator loss.
    - VAEs and GANs are "constructive", so if I want a set of dog pictures, they will generate images of dogs based on a sample set provided.
    - DCGAN is much more robust, being able to function on train-test subsets of different datasets (CIFAR-10 vs Imagenet 1k).
    - Stride is better in GANs than it is in CNNs, as it is faster and cheaper.
## What’s next on the list:
- Continue working on the model.
    - Find a way to process the images. Convert from image format (nxmx3) to a n/2 x m/2 x 1 array.
    - Use one-hot encoding to convert labels to a n x 62 array (n is the number of letters, between 4 and 6).
    - Graph loss and accuracy of the metrics, possibly visualize
    - Implement load/save as well as non-dataset testing of model.
## Interesting tidbits/thoughts:
- DCGAN isn't so much a type of GAN but rather a foundationally different type of GAN, using deep CNNs in the architecture to test its ability in ULearning.
    - Interestingly, many of the more notable GAN papers following DCGAN are derivatives of it. 
    - Provided basis of accurate model training and archiecture that other papers later built on
- Huh, never knew that GANs used CNN in their architecture, but I always kind of suspected that they were at play, since CNNs are just too good at identifying images.
## Data/Resources
- https://arxiv.org/abs/1511.06434: The DCGAN paper, which outlines the use of deep CNNs as a way to converge better and produce cleaner output
- http://man.hubwiz.com/docset/TensorFlow.docset/Contents/Resources/Documents/api_docs/python/tf/keras/layers/Flatten.html: Flatten() keras/TF documentation
- https://keras.io/api/layers/regularization_layers/dropout/: Dropout layer Keras/TF documentation
