# Convolutional Network on MNIST dataset

## Usage<br>

**Step 1: Installation** <br>
- Download [Anaconda](https://www.anaconda.com/download/)
- Install [tensorflow / tensorflow-gpu](https://www.tensorflow.org/install/) 
- Install [**keras**](https://keras.io/#installation)


**Step 2: Download Files**<br>
Clone this repo and [Download](https://github.com/iphton/Kaggle-Competition/tree/gh-pages/Digit%20Recognizer) all files save it
on working directory.

**Step 3: Open Jupyter Notebook**<br>
Open the notebook and navigate to the downloaded folder and search for `MNIST_Keras.ipynb`. After opening the 
nodebook, just a run each cell consecutively. 


## Computational Graph<br>
Open up `CMD` and run folowing command: `tensorboard --logdir ConvNet/KerasGraph`. You should get a locahost address, just copy and 
paste it on web browser. Alternatively, after running above command and started the server, go to any browser and type `localhost:(port-number)`.

![model_graph](https://user-images.githubusercontent.com/17668390/46923691-5e641580-d03d-11e8-91d2-2a1049fb7ce4.png)

## Results<br>
I got **Final loss: 0.01153, Final accuracy: 0.99595** with implementing **ConvNet** using TensorFlow high level API **Keras** on **GeForce GTX 1050 Ti**. 

I set several number of epochs (20 , 50 , 100) on the training process. However, we can easily get almost **99%+** accuracy within 20 or 30 epochs. Training on a single CPU, epochs size should be set within 2 or 3~4, accuracy almost **99%**.

*Accuracy:*

![accuracy](https://user-images.githubusercontent.com/17668390/46923669-05947d00-d03d-11e8-99d4-fc9f29e607e3.PNG)

*Loss:*

![loss](https://user-images.githubusercontent.com/17668390/46923671-13e29900-d03d-11e8-89b3-4ff3c38283c2.PNG)

*Validation Accuracy:*

![validation_acc](https://user-images.githubusercontent.com/17668390/46923676-28269600-d03d-11e8-9e31-d1bd78fbe564.PNG)

*Validation Loss:*

![validation_loss](https://user-images.githubusercontent.com/17668390/46923681-35dc1b80-d03d-11e8-82c6-a8982c372e4f.PNG)
