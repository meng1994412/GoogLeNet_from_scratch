# GoogLeNet from Scratch
## Objectives
Implement GoogLeNet family from scratch, including MiniGoogLeNet and GoogLeNet, and train them on CIFAR-10, Tiny ImageNet, and ImageNet datasets.
* Construct MiniGoogLeNet and train the network on CIFAR-10 datasets to obtain ≥90% accuracy.
* Construct GoogLeNet and train the network on Tiny ImageNet Visual Recognition Challenge and claim a top ranking position on Leaderboard.

## Packages Used
* Python 3.6
* [OpenCV](https://docs.opencv.org/3.4.4/) 4.0.0
* [keras](https://keras.io/) 2.2.4 for GoogLeNet on CIFAR-10 and 2.1.0 for the rest
* [Tensorflow](https://www.tensorflow.org/install/) 1.13.0
* [cuda toolkit](https://developer.nvidia.com/cuda-toolkit) 10.0
* [cuDNN](https://developer.nvidia.com/cudnn) 7.4.2
* [scikit-learn](https://scikit-learn.org/stable/) 0.20.2
* [Imutils](https://github.com/jrosebr1/imutils)
* [NumPy](http://www.numpy.org/)

## Approaches
### MiniGoogLeNet on CIFAR-10
The details about CIFAR-10 datasets can be found [here](https://www.cs.toronto.edu/~kriz/cifar.html).

The MiniGoogLeNet on CAFAR-10 dataset is inspired by [Eric Jang](https://twitter.com/ericjang11) and [pluskid](https://twitter.com/pluskid).

There are three modules inside MiniGoogLeNet, including Conv module, inception module, downsample module. Figure 1 shows the MiniGoogLeNet architecture.

<img src="https://github.com/meng1994412/GoogLeNet_from_scratch/blob/master/output/minigooglenet_architecture.png" width="600">

Figure 1: MiniGoogLeNet architecture ([reference](https://arxiv.org/abs/1611.03530)).

The MiniGoogLeNet architecture can be found in `minigooglenet.py` ([check here](https://github.com/meng1994412/GoogLeNet_from_scratch/blob/master/pipeline/nn/conv/minigooglenet.py)) under `pipeline/nn/conv/` directory. The input to the model includes dimensions of the image (height, width, depth, and number of classes). In this part, the input would be (width = 32, height = 32, depth = 3, classes = 10).

The `googlenet_cifar10.py` ([check here](https://github.com/meng1994412/GoogLeNet_from_scratch/blob/master/googlenet_cifar10.py)) is responsible for training the network, evaluating the model (including plotting the loss and accuracy curve of training and validation sets, providing the classification report), and serialize the model to disk.

There is a helper class:

The `trainingmonitor.py` ([check here](https://github.com/meng1994412/GoogLeNet_from_scratch/blob/master/pipeline/callbacks/trainingmonitor.py)) under `pipeline/callbacks/` directory create a `TrainingMonitor` callback that will be called at the end of every epoch when training a network. The monitor will construct a plot of training loss and accuracy. Applying such callback during training will enable us to babysit the training process and spot overfitting early, allowing us to abort the experiment and continue trying to tune parameters.

We could use following command to train the model.

```
python googlenet_cifar10.py --model output/minigooglenet_cifar10.hdf5 --output output
```

### GoogLeNet on Tiny ImageNet Visual Recognition Challenge
The details about the challenge and dataset can be found [here](https://tiny-imagenet.herokuapp.com/).

The `tiny_imagenet_config.py` ([check here](https://github.com/meng1994412/GoogLeNet_from_scratch/blob/master/config/tiny_imagenet_config.py)) under `config/` directory stores all relevant configurations for the project, including the paths to input images, total number of class labels, information on the training, validation, and testing splits, path to the HDF5 datasets, and path to output models, plots, and etc.

#### Build the infrastructure for `HDF5` dataset
The `hdf5datasetwriter.py` ([check here](https://github.com/meng1994412/GoogLeNet_from_scratch/blob/master/pipeline/io/hdf5datasetwriter.py)) under `pipeline/io/` directory, defines a class that help to write raw images or features into `HDF5` dataset.

The `build_tiny_imagenet.py` ([check here](https://github.com/meng1994412/GoogLeNet_from_scratch/blob/master/build_tiny_imagenet.py)) is used for serializing the raw images into an `HDF5` dataset. Although `Keras` has methods that can allow us to use the raw file paths on disk as input to the training process, this method is highly inefficient. Each and every image residing on disk requires an I/O operation which introduces latency into training pipeline. Not only is `HDF5` capable of storing massive dataset, but it is optimized for I/O operations.

We could use following command to build Tiny ImageNet dataset.

```
python build_tiny_imagenet.py
```

#### Build image pre-processors
The `meanpreprocessor.py` ([check here](https://github.com/meng1994412/GoogLeNet_from_scratch/blob/master/pipeline/preprocessing/meanpreprocessor.py)) under `pipeline/preprocessing/` directory subtracts the mean red, green, and blue pixel intensties across the training set, which is a form of data normalization. Mean subtraction is used to reduce the affects of lighting variations during classification.

The `simplepreprocessor.py` ([check here](https://github.com/meng1994412/GoogLeNet_from_scratch/blob/master/pipeline/preprocessing/simplepreprocessor.py)) under `pipeline/preprocessing/` directory defines a class to change the size of image. This class is just used to ensure that each input image has dimenison of 64x64x3.

The `imagetoarraypreprocessor.py` ([check here](https://github.com/meng1994412/GoogLeNet_from_scratch/blob/master/pipeline/preprocessing/imagetoarraypreprocessor.py)) under `pipeline/preprocessing/` directory defines a class to convert the image dataset into keras-compatile arrays.

#### Construct GoogLeNet architecture from scratch
Figure 2 shows the micro-architecture of inception module in GoogLeNet.

<img src="https://github.com/meng1994412/GoogLeNet_from_scratch/blob/master/output/inception_module.png" width="500">

Figure 2: Inception module, which is the micro architecture of the GoogLeNet ([reference](https://arxiv.org/abs/1409.4842)).

Table 1 illustrates the GoogLeNet architecture ([reference](https://arxiv.org/abs/1409.4842)).

| layer type | patch size/stride | output size | depth | #1x1 | #3x3 reduce| #3x3 | #5x5 reduce | #5x5 | pool proj |
| ------------- |:-----:| :---------:|:-:|:---:|:---:|:---:|:---:|:---:|:---:|
| convolution   | 7x7/2 | 112x112x64 | 1 |     |     |     |     |     |     |
| max pool      | 3x3/2 |  56x56x64  | 0 |     |     |     |     |     |     |
| convolution   | 3x3/1 | 56x56x192  | 2 |     | 64  | 192 |     |     |     |
| max pool      | 3x3/2 | 28x28x192  | 0 |     |     |     |     |     |     |
| inception(3a) |       | 28x28x256  | 2 | 64  | 96  | 128 | 16  | 32  | 32  |
| inception(3b) |       | 28x28x480  | 2 | 128 | 128 | 192 | 32  | 96  | 64  |
| max pool      | 3x3/2 | 14x14x480  | 0 |     |     |     |     |     |     |
| inception(4a) |       | 14x14x512  | 2 | 192 | 96  | 208 | 16  | 48  | 64  |
| inception(4b) |       | 14x14x512  | 2 | 160 | 112 | 224 | 24  | 64  | 64  |
| inception(4c) |       | 14x14x512  | 2 | 128 | 128 | 256 | 24  | 64  | 64  |
| inception(4d) |       | 14x14x528  | 2 | 112 | 144 | 288 | 32  | 64  | 64  |
| inception(4e) |       | 14x14x832  | 2 | 256 | 160 | 320 | 32  | 128 | 128 |
| max pool      | 3x3/2 |  7x7x832   | 0 |     |     |     |     |     |     |
| inception(5a) |       |  7x7x832   | 2 | 256 | 160 | 320 | 32  | 128 | 128 |
| inception(5b) |       |  7x7x1024  | 2 | 384 | 192 | 384 | 48  | 128 | 128 |
| avg pool      | 7x7/1 |  1x1x1024  | 0 |     |     |     |     |     |     |
| dropout(40%)  |       |  1x1x1024  | 0 |     |     |     |     |     |     |
| linear        |       |  1x1x1000  | 1 |     |     |     |     |     |     |
| softmax       |       |  1x1x1000  | 0 |     |     |     |     |     |     |

Instead of using 7x7 filters with stride of 2x2 in the first convolution layer, I use 5x5 filters with stride of 1x1, since the input images have dimension of 64x64x3, unlike original GoogLeNet which has input dimension of 224x224x3. Thus, 7x7 filters with stride of 2x2 will reduce the dimensions too quickly. Also the size of average pooling layer should be 4x4 instead of 7x7, with stride of 1.

The GoogleNet can be found in `googlenet.py` ([check here](https://github.com/meng1994412/GoogLeNet_from_scratch/blob/master/pipeline/nn/conv/googlenet.py)) under `nn/conv/` directory.

#### Train the GoogLeNet and evaluate it
I use a "ctrl+c" method to train the model as a baseline. By using this method, I can start training with an initial learning rate (and associated set of hyperparameters), monitor training, and quickly adjust the learning rate based on the results as they come in.

The `train.py` ([check here](https://github.com/meng1994412/GoogLeNet_from_scratch/blob/master/train.py)) is responsible for training the baseline model. The `TrainingMonitor` callback is responsible for plotting the loss and accuracy curves of training and validation sets. And the `EpochCheckpoint` callback is responsible for saving the model every 5 epochs.

After getting a sense of baseline model, I will switch to use method of learning rate decay to re-train the model. The `train_decay.py` ([check here](https://github.com/meng1994412/GoogLeNet_from_scratch/blob/master/train_decay.py)) change the method from "ctrl+c" to learning rate decay to re-train the model. The `TrainingMonitor` callback again is responsible for plotting the loss and accuracy curves of training and validation sets. The `LearningRateScheduler` callback is responsible for learning rate decay.

The `rank_accuracy.py` ([check here](https://github.com/meng1994412/GoogLeNet_from_scratch/blob/master/rank_accuracy.py)) measures the `rank-1` and `rank-5` accuracy of the model by using the testing set.

There are some helper classes for training process, including:

The `EpochCheckpoint.py` ([check here](https://github.com/meng1994412/GoogLeNet_from_scratch/blob/master/pipeline/callbacks/epochcheckpoint.py)) under `pipeline/callbacks/` directory can help to store individual checkpoints for GoogLeNet so that we do not have to retrain the network from beginning. The model is stored every 5 epochs.

The `hdf5datasetgenerator.py` ([check here](https://github.com/meng1994412/GoogLeNet_from_scratch/blob/master/pipeline/io/hdf5datasetgenerator.py)) under `pipeline/io/` directory yields batches of images and labels from `HDF5` dataset. This class can help to facilitate our ability to work with datasets that are too big to fit into memory.

The `ranked.py` ([check here](https://github.com/meng1994412/GoogLeNet_from_scratch/blob/master/pipeline/utils/ranked.py)) under `pipeline/utils/` directory contains a helper function to measure both the `rank-1` and `rank-5` accuracy when the model is evaluated by using testing set.

We could use following command to train the model if we start from the beginning.
```
python train.py --checkpoints output/checkpoints
```

If we start the training at middle of the epochs (simply use a number to replace `{epoch_number_you_want_to_start}`):
```
python train.py --checkpoints output/checkpoints --model output/checkpoints/epoch_{epoch_number_you_want_to_start}.hdf --start_epoch {the_epoch_number_you_want_to_start}
```

For learning rate decay, just use following command:
```
python train_decay.py --model output/googlenet_tinyimagenet_decay.hdf5
```

In order to use testing set to evaluate the network, use the following command:
```
python rank_accuracy.py
```

## Results
### MiniGoogLeNet on CIFAR-10
Figure 3 demonstrates the loss and accuracy curve of training and validation sets. And Figure 4 shows the evaluation of the network, which indicate a 90% accuracy.

<img src="https://github.com/meng1994412/GoogLeNet_from_scratch/blob/master/output/7879.png" width="500">

Figure 3: Plot of training and validation loss and accuracy.

<img src="https://github.com/meng1994412/GoogLeNet_from_scratch/blob/master/output/minigooglenet_evaluation.png" width="400">

Figure 4: Evaluation of the network, indicating 90% accuracy.

### GoogLeNet on Tiny ImageNet
#### Experiment 1
In experiment 1, I use "ctrl+c" method with learning rate schedule shown as Table 2. `SGD` optimizer with momentum of 0.9 and nesterov acceleration is used. The sequence of `convolution_module` is `CONV => BN => ReLU`

Table 2: Learning rate schedule for experiment 1.

| Epoch | Learning Rate |
|:-----:|:-------------:|
|1 - 40 | 1e-3          |
|41 - 60| 1e-4          |
|61 - 70| 1e-5          |

Figure 5 demonstrates the loss and accuracy curve of training and validation sets. And Figure 6 shows the evaluation of the network, which indicate 55.05% rank-1 accuracy and 79.64% rank-5 accuracy.

<img src="https://github.com/meng1994412/GoogLeNet_from_scratch/blob/master/output/googlenet_tinyimagenet_1.png" width="500">

Figure 5: Plot of training and validation loss and accuracy.

<img src="https://github.com/meng1994412/GoogLeNet_from_scratch/blob/master/output/googlenet_tiny_imagenet_1.png" width="200">

Figure 6: Evaluation of the network, indicating 55.05% rank-1 accuracy and 79.64% rank-5 accuracy.

#### Experiment 2
In experiment 2, I still use learning rate in Table 2. In order to increase the accuracy, I change the `convolution module` to use `CONV => RELU => BN` sequence instead of `CONV => BN => RELU`.

Figure 7 demonstrates the loss and accuracy curve of training and validation sets. And Figure 8 shows the evaluation of the network, which indicate 55.41% rank-1 accuracy and 80.68% rank-5 accuracy. There is about 0.35% increment in rank-1 accuracy and 1% increment in rank-5 accuracy, comparing to the result in experiment 1.

<img src="https://github.com/meng1994412/GoogLeNet_from_scratch/blob/master/output/googlenet_tinyimagenet_2.png" width="500">

Figure 7: Plot of training and validation loss and accuracy.

<img src="https://github.com/meng1994412/GoogLeNet_from_scratch/blob/master/output/googlenet_tiny_imagenet_2.png" width="200">

Figure 8: Evaluation of the network, indicating 55.41% rank-1 accuracy and 80.68% rank-5 accuracy.

#### Experiment 3
In the experiment 3, I use the `convolution module` in [experiment 2](#experiment-2), but change the method from "ctrl+c" to learning rate decay. And the number of epoch is still 70.

Figure 9 demonstrates the loss and accuracy curve of training and validation sets. And Figure 10 shows the evaluation of the network, which indicate 57.34% rank-1 accuracy and 81.25% rank-5 accuracy.

<img src="https://github.com/meng1994412/GoogLeNet_from_scratch/blob/master/output/googlenet_tinyimagenet_3.png" width="500">

Figure 9: Plot of training and validation loss and accuracy.

<img src="https://github.com/meng1994412/GoogLeNet_from_scratch/blob/master/output/googlenet_tiny_imagenet_3.png" width="200">

Figure 10: Evaluation of the network, indicating 57.34% rank-1 accuracy and 81.25% rank-5 accuracy.

By using this rank-1 accuracy, I can claim #5 position on the Leaderboard in [Tiny ImageNet Visual Recognition Challenge](https://tiny-imagenet.herokuapp.com/).
