# GoogLeNet from Scratch
## Objectives
Implement GoogLeNet family from scratch, including MiniGoogLeNet and GoogLeNet, and train them on CIFAR-10, Tiny ImageNet, and ImageNet datasets.
* Construct MiniGoogLeNet and train the network on CIFAR-10 datasets to obtain ≥90% accuracy.
* Construct GoogLeNet and train the network on Tiny ImageNet Visual Recognition Challenge and claim a top ranking position on Leaderboard

## Packages Used
* Python 3.6
* [OpenCV](https://docs.opencv.org/3.4.4/) 3.4.4
* [keras](https://keras.io/) 2.2.4
* [Tensorflow](https://www.tensorflow.org/install/) 1.12.0
* [cuda toolkit](https://developer.nvidia.com/cuda-toolkit) 9.0
* [cuDNN](https://developer.nvidia.com/cudnn) 7.1.2
* [scikit-learn](https://scikit-learn.org/stable/) 0.20.2
* [Imutils](https://github.com/jrosebr1/imutils)
* [NumPy](http://www.numpy.org/)

## Approaches
### MiniGoogLeNet on CIFAR-10
The MiniGoogLeNet on CAFAR-10 dataset is inspired by [Eric Jang](https://twitter.com/ericjang11) and [pluskid](https://twitter.com/pluskid).

There are three modules inside MiniGoogLeNet, including Conv module, inception module, downsample module. Figure 1 shows the MiniGoogLeNet architecture.

<img src="https://github.com/meng1994412/GoogLeNet_from_scratch/blob/master/output/minigooglenet_architecture.png" width="600">

Figure 1: MiniGoogLeNet architecture ([reference](https://arxiv.org/pdf/1611.03530.pdf)).

The MiniGoogLeNet architecture can be found in `minigooglenet.py` ([check here](https://github.com/meng1994412/GoogLeNet_from_scratch/blob/master/pipeline/nn/conv/minigooglenet.py)) under `pipeline/nn/conv/` directory. The input to the model includes dimensions of the image (height, width, depth, and number of classes). In this part, the input would be (width = 32, height = 32, depth = 3, classes = 10).

The `googlenet_cifar10.py` ([check here](https://github.com/meng1994412/GoogLeNet_from_scratch/blob/master/googlenet_cifar10.py)) is responsible for training the network, evaluating the model (including plotting the loss and accuracy curve of training and validation sets, providing the classification report), and serialize the model to disk.

The `trainingmonitor.py` ([check here](https://github.com/meng1994412/GoogLeNet_from_scratch/blob/master/pipeline/callbacks/trainingmonitor.py)) under `pipeline/callbacks/` directory create a `TrainingMonitor` callback that will be called at the end of every epoch when training a network. The monitor will construct a plot of training loss and accuracy. Applying such callback during training will enable us to babysit the training process and spot overfitting early, allowing us to abort the experiment and continue trying to tune parameters.

## Results
### MiniGoogLeNet on CIFAR-10
Figure 2 demonstrates the loss and accuracy curve of training and validation sets. And Figure 3 shows the evaluation of the network, which indicate a 90% accuracy.

<img src="https://github.com/meng1994412/GoogLeNet_from_scratch/blob/master/output/7879.png" width="500">

Figure 2: Plot of training and validation loss and accuracy.

<img src="https://github.com/meng1994412/GoogLeNet_from_scratch/blob/master/output/minigooglenet_evaluation.png" width="400">

Figure 3: Evaluation of the network, indicating 90% accuracy.
