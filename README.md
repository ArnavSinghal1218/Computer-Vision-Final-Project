# Explainable emotion detection for facial expressions using Bcos networks on the FER2013 dataset

This repository contains the implementation of Emotion Detection using two different Convolutional Neural Network (CNN) architectures: a standard CNN and a CNN enhanced with B-cos layers. The goal of this project is to explore the performance and interpretability of the B-cos CNN in comparison to a traditional CNN for detecting emotions from facial expressions using the FER2013 dataset.

## Table of Contents
- Overview

- Dataset

- CNN Architecture

- B-cos Network Architecture

- Approach

- Experiments

- Results

- Conclusion

- References

## Overview
Deep learning models like CNNs have shown significant promise in emotion recognition, especially when dealing with facial expression data. However, traditional CNNs often function as "black boxes" without providing clear insights into their decision-making processes. This project explores a novel B-cos CNN that integrates interpretability directly into the model, providing more transparent insights into how the network recognizes emotions from facial features.

## Dataset
We use the FER2013 dataset, which contains 48x48 grayscale images of human faces, each labeled with one of the seven emotion classes:

- Angry
- Disgust
- Fear
-Happy
- Sad
- Surprise
- Neutral

The dataset consists of 28,709 training images and 7,178 test images.

## CNN Architecture
The baseline model is a traditional deep CNN consisting of multiple layers of convolution and pooling operations, followed by fully connected layers for emotion classification. This standard architecture is effective in learning spatial hierarchies from pixel data but lacks built-in interpretability.

### Model Flow:
1. Convolutional Layers: Extract features like edges and textures from the input image.
2. Pooling Layers: Reduce the spatial dimensions of the image while preserving important features.
3. Fully Connected Layers: Convert feature maps into probability distributions over the seven emotion classes.

## B-cos Network Architecture
The B-cos network modifies the traditional CNN architecture by incorporating B-cos layers, which enhance the interpretability of the convolution operations. These layers provide meaningful gradients that highlight the critical regions in the input image (e.g., eyes, mouth) that influence the network's decision, making it easier to understand the model’s classification process.

### Benefits of B-cos Layers:
- Built-in interpretability without the need for post-hoc methods.
- Visual gradient maps highlight the key facial features used for emotion detection.
- More focused decision-making compared to traditional CNNs.

## Approach
### Standard CNN:
- Optimizer: Stochastic Gradient Descent (SGD) with momentum.
- Loss Function: Cross-entropy loss.
- Data Augmentation: Horizontal flips, normalization.
- Training: Multiple epochs with learning rate scheduling and batch size of 64.
### B-cos CNN:
- Same training process and hyperparameters as the standard CNN.
- The B-cos layers alter the way features are learned by focusing on interpretability.

Both models were trained and validated on the FER2013 dataset, with performance metrics such as accuracy, loss, and interpretability visualizations compared.

## Experiments
Several experiments were conducted to evaluate both models:

- Training & Validation Accuracy: Performance of both models was tracked across multiple epochs to monitor convergence and generalization.
- Confusion Matrix: Provided detailed performance insights across the seven emotion classes.
- Gradient Visualizations: Used to understand which regions of the face the models focused on while making predictions. The CNNBcos model provided clearer and more interpretable gradient visualizations than the standard CNN.

### Data Preprocessing:
- Rescaling: Image pixel values rescaled to [0, 1].
- Augmentation: Random horizontal flips and normalization to improve model generalization.

### Key Metrics:
Accuracy: Both models achieved similar accuracy on the FER2013 dataset.
Interpretability: The B-cos CNN outperformed the standard CNN in terms of transparency, focusing on key facial regions during emotion recognition.

## Results
- Performance: Both CNN and CNNBcos achieved comparable accuracy in emotion classification, with minor performance improvements in certain emotions (e.g., happiness, anger) for the CNNBcos model.
- Interpretability: The CNNBcos model showed significantly better gradient visualizations, highlighting critical facial areas (such as eyes and mouth) that are crucial for identifying emotions. The traditional CNN, by contrast, provided more diffuse, less intuitive heatmaps.

## onclusion
The CNNBcos model provides a viable solution to the interpretability issues of traditional CNNs. Without sacrificing classification accuracy, the B-cos CNN enhances transparency, making it easier to understand how the model arrives at its decisions. This feature is essential in domains where trust and explainability are critical, such as healthcare and human-computer interaction.

By incorporating B-cos layers, we can ensure that deep learning models not only perform well but also provide insights into their decision-making processes.

## References
1. Mollahosseini, A., Hasani, B., & Mahoor, M. H. (2017). AffectNet: A Database for Facial Expression, Valence, and Arousal Computing in the Wild. IEEE Transactions on Affective Computing.
2. Zhang, Z., & Zhang, L. (2017). A CNN-Based Approach for the Automated Emotion Recognition in Interaction Process. IEEE International Conference on Information Technology.
3. Montavon, G., Lapuschkin, S., Binder, A., Samek, W., & Müller, K. (2017). Explaining Nonlinear Classifiers Decisions with Deep Taylor Decomposition. Pattern Recognition, 65, 211-222.
4. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why Should I Trust You?": Explaining the Predictions of Any Classifier. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.
5. Simonyan, K., Vedaldi, A., & Zisserman, A. (2014). Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps. arXiv preprint arXiv:1312.6034.
6. [B¨ohle et al., 2022] B¨ohle, M., Fritz, M., and Schiele, B. (2022). B-cos networks: Align-ment is all we need for interpretability. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10329–10338.
7. [Khaireddin and Chen, 2021] Khaireddin, Y. and Chen, Z. (2021). Facial emotion recognition: State of the art performance on fer2013. arXiv preprint arXiv:2105.03588.
