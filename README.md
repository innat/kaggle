# Kaggle Competition
- ✔ [Titanic: Machine Learning from Disaster](http://nbviewer.jupyter.org/github/iphton/Kaggle-Competition/blob/gh-pages/Titanic%20Competition/Notebook/Predict%20survival%20on%20the%20Titanic.ipynb)<br>
In this kaggle challenge, we're asked to complete the analysis of what sorts of people were likely to survive. In particular, we're asked to apply the tools of machine learning to predict which passengers survived the tragedy. I compared [10 popular classifiers](http://nbviewer.jupyter.org/github/iphton/Kaggle-Competition/blob/gh-pages/Titanic%20Competition/Notebook/Predict%20survival%20on%20the%20Titanic.ipynb#10-bullet) and evaluate the mean accuracy of each of them by a stratified kfold cross validation procedure. And finally explore following models and fine-tune each separately:
  - GB Classifier
  - Linear Discriminant Analysis
  - Logistic Regression
  - Random Forest Classifer
  - Support Vectore Machine
  
  Finally, We've used **voting classifier** to combine the predictions coming from the 5 classifiers. And got prediction accuracy almost **82.97%** 

- ✔ [Digit Classification](https://github.com/iphton/Kaggle-Competition/tree/gh-pages/Digit%20Recognizer)<br>
The goal in this competition is to take an image of a handwritten single digit, and determine what that digit is. For every **ImageId** in the test set, we should predict the correct label. We've implemented two architectures stated following and both has achieved **99%+** accuracy on the test set. 
  - Capsule Network or [**CapsNets**](https://github.com/iphton/Kaggle-Competition/tree/gh-pages/Digit%20Recognizer/CapsuleNet) are a new neural net architecture that may well have a profound impact on deep learning, in particular for computer vision. **Geoffrey Hinton** proposed this architecture that introduced a completely new type of neural network based on so-called **capsules**. Geoffrey Hinton - [Paper: Dynamic Routing Between Capsules](https://arxiv.org/abs/1710.09829). I found [this](https://www.oreilly.com/ideas/introducing-capsule-networks) blog post by **Aurélien Géron** well explained on this topic. However, most of the implemented function to build **CapsuleNet** is adopted from [Xifeng Guo Ph.D.](https://github.com/XifengGuo).
  
  - Convolutional Network or [**ConvNets**](https://github.com/iphton/Kaggle-Competition/tree/gh-pages/Digit%20Recognizer/ConvNet) - one of the most profoundly used feed forward neural net for image classification and massively well known for it's various types of implementation in computer vision spheres such as *AlexNet*, *VGGNet*, *ResNet*, *Xception*, *Inception* etc. Back in 2012 **AlexNet** competed in the *ImageNet Large Scale Visual Recognition Challenge*. It has had a large impact on the field of machine learning, specifically in the application of deep learning to machine vision. 
