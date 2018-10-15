# Capsule Network on MNIST dataset

## Usage<br>

**Step 1: Installation** <br>
- Download [Anaconda](https://www.anaconda.com/download/)
- Install [tensorflow / tensorflow-gpu](https://www.tensorflow.org/install/) 
- Install [**keras**](https://keras.io/#installation)


**Step 2: Download Files**<br>
Clone this repo and [Download](https://github.com/iphton/Kaggle-Competition/tree/gh-pages/Digit%20Recognizer) dataset files and save it
on working directory.

**Step 3: Open Jupyter Notebook**<br>
Open the notebook and navigate to the downloaded folder and search for `MNIST_CapsuleNet.ipynb`. After opening the 
nodebook, just a run each cell consecutively. 


## Computational Graph | CapsNet<br>
Open up `CMD` and run folowing command: `tensorboard --logdir ConvNet/KerasGraph`. You should get a localhost address, just copy and 
paste it on web browser. Alternatively, after running above command and started the server, go to any browser and type `localhost:(port-number)`.

![computation_graph](https://user-images.githubusercontent.com/17668390/46923780-dbdc5580-d03e-11e8-85ae-39f92e2dcb88.png)

## Results<br>

**Test accuracy achieves 99%.** The computational process is pretty complex and training **CapsNet** by setting epoch at too high on weak 
machine may hang easily. I ran the **CpasNet** model with GeForce *GTX 1050 Ti* at **10 epoch** and it took me almost 40 mins. 

Accuracy:

![accuracy](https://user-images.githubusercontent.com/17668390/46923787-ec8ccb80-d03e-11e8-993d-d6ec72b2f0ec.PNG)

Loss:

![loss](https://user-images.githubusercontent.com/17668390/46923791-f6aeca00-d03e-11e8-8691-05f21cf0aec7.PNG)

Validation Accuracy:

![validation_acc](https://user-images.githubusercontent.com/17668390/46923676-28269600-d03d-11e8-9e31-d1bd78fbe564.PNG)

Validation Loss:

![validation_loss](https://user-images.githubusercontent.com/17668390/46923799-0a5a3080-d03f-11e8-9bb7-fd7ac1082818.PNG)

Learning Rate Decay:

![learning_rate](https://user-images.githubusercontent.com/17668390/46923803-1c3bd380-d03f-11e8-8276-d8ab6984f55b.PNG)

<br>

## Miscellaneous Implementations

- PyTorch:
  - [XifengGuo/CapsNet-Pytorch](https://github.com/XifengGuo/CapsNet-Pytorch)
  - [timomernick/pytorch-capsule](https://github.com/timomernick/pytorch-capsule)
  - [gram-ai/capsule-networks](https://github.com/gram-ai/capsule-networks)
  - [nishnik/CapsNet-PyTorch](https://github.com/nishnik/CapsNet-PyTorch.git)
  - [leftthomas/CapsNet](https://github.com/leftthomas/CapsNet)
  
- TensorFlow:
  - [naturomics/CapsNet-Tensorflow](https://github.com/naturomics/CapsNet-Tensorflow.git)   
  I referred to some functions in this repository.
  - [InnerPeace-Wu/CapsNet-tensorflow](https://github.com/InnerPeace-Wu/CapsNet-tensorflow)   
  - [chrislybaer/capsules-tensorflow](https://github.com/chrislybaer/capsules-tensorflow)

- MXNet:
  - [AaronLeong/CapsNet_Mxnet](https://github.com/AaronLeong/CapsNet_Mxnet)
  
- Chainer:
  - [soskek/dynamic_routing_between_capsules](https://github.com/soskek/dynamic_routing_between_capsules)

- Matlab:
  - [yechengxi/LightCapsNet](https://github.com/yechengxi/LightCapsNet)
