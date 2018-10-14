## Hand Written Digit Classification

The data files `train.csv` and `test.csv` contain gray-scale images of hand-drawn digits, from zero through nine. Each image is 28 pixels 
in height and 28 pixels in width, for a total of 784 pixels in total. Each pixel has a single pixel-value associated with it, indicating 
the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255, inclusive.

The training data set, (train.csv), has 785 columns. The first column, called label, is the digit that was drawn by the user. The rest of
the columns contain the pixel-values of the associated image.

The test data set, (test.csv), is the same as the training set, except that it does not contain the label column.

## Task<br>
Goal is to correctly identify digits from a dataset of tens of thousands of handwritten images. 

## Deep Learninig Models<br>
To accomplish this task we've implemented two neural architectures. Now, MNIST ("Modified National Institute of Standards and Technology") is the de facto “hello world” dataset of computer vision. As new machine learning techniques emerge, MNIST remains a reliable resource for researchers and learners alike. 

We've worked on a well known archtecture which is **Convolutional Neural Network**. The one of the most profoundly used neural network for computer vision in deep learning spheres. But it also has some drawbacks. **Geoffrey Hinton** proposed a architecture that introduced a completely new type of neural network based on so-called capsules. **CapsNets** are a new neural net architecture that may well have a profound impact on deep learning, in particular for computer vision. [Read](https://arxiv.org/abs/1710.09829) the paper of Hinton working on *Dynamic Routing Between Capsules*.

We've implemented both archtecture and both has achieved **99%** accuracy on the test set. One of the most exciting things that I've found is that, the accuracy of the models in both case are significantly higher on the test set which means our models isn't overfitting. 

- Convolutional Network
- Capsule Network

However, at the moment **CapsNets** is still under development. To get the intuition of this network **Aurélien Géron** ( one of my favorite person ) made a awesome [intro video](https://www.youtube.com/watch?v=pPN8d0E3900) on it.

## About this repositories<br>
This is nothing like serious project. Just curiously wanted to see the working procedure of **CapsNet** archtecture on image classification and yes, I traditionally choose **MNIST** dataset.

## Usages<br>
Right now, there're two folders. Namely,

- ConvNet
- CapsuleNet

In **ConvNet**, we've implemented Convolutional network using **Tensorflow's** high level API **Keras** by creating several convolution layers. We used **Adam** for the optimization of cost function. After training the model, we save the **model** and **weights** in `json` and `hdf5` format respectively on disk. We also saw the *learning curve* by calling the **tensorboard** and viz the computatinal graph as well. However, all the set-up and execute instruction of this model is clearly provided on the jupyter notebook, in place inside the **ConvNet** folder.

And the **CapsNet**, the point of interest of the work. It's so heavy to run the model. I trained the model on **GeForce GTX 1050 Ti**. I set **10** epochs on the training process and took almost 40 mins. However, Training on a single CPU, epochs size should be set within < 3. The computation task is pretty complex and takes lots of computation power. We saw the *learning curve* by calling the **tensorboard** and viz the computatinal graph as well. However, all the set-up and execute instruction of this model is clearly provided on the jupyter notebook, in place inside the **CapsNet** folder.

