# set matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import packages
from config import tiny_imagenet_config as config
from pipeline.preprocessing import ImageToArrayPreprocessor
from pipeline.preprocessing import SimplePreprocessor
from pipeline.preprocessing import MeanPreprocessor
from pipeline.callbacks import TrainingMonitor
from pipeline.callbacks import EpochCheckpoint
from pipeline.io import HDF5DatasetGenerator
from pipeline.nn.conv import GoogLeNet
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.optimizers import Adam
from keras.models import load_model
from keras import backend as K
import argparse
import json

# define the total number of epochs to train for along with the initial learning rate
NUM_EPOCHS = 75
INIT_LR = 1e-3

def poly_decay(epoch):
    # initialize the maximum number of epochs, base learning rate,
    # and power of the polynomial
    maxEpochs = NUM_EPOCHS
    baseLR = INIT_LR
    power = 1.0

    # compute the new learning rate based on polynomial decay
    alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power

    # return the new learning rate
    return alpha

# construct argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required = True,
    help = "path to output model")
args = vars(ap.parse_args())

# construct the training image generator for data augmentation
aug = ImageDataGenerator(rotation_range = 18, zoom_range = 0.15,
    width_shift_range = 0.2, height_shift_range = 0.2, shear_range = 0.15,
    horizontal_flip = True, fill_mode = "nearest")

# load the RGB means for the training set
means = json.loads(open(config.DATASET_MEAN).read())

# initialize the image preprocessors
sp = SimplePreprocessor(64, 64)
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
iap = ImageToArrayPreprocessor()

# initialize the training and validation dataset generators
trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, 64, aug = aug,
    preprocessors = [sp, mp, iap], classes = config.NUM_CLASSES)
valGen = HDF5DatasetGenerator(config.VAL_HDF5, 64,
    preprocessors = [sp, mp, iap], classes = config.NUM_CLASSES)

# initialize the optimizer and model
print("[INFO] compiling model...")
model = GoogLeNet.build(width = 64, height = 64, depth = 3,
    classes = config.NUM_CLASSES, reg = 0.0002)
opt = Adam(lr = INIT_LR)
model.compile(loss = "categorical_crossentropy", optimizer = opt,
    metrics = ["accuracy"])

# construct the set of callbacks
callbacks = [
    TrainingMonitor(config.FIG_PATH, jsonPath = config.JSON_PATH),
    LearningRateScheduler(poly_decay)
]

# train the network
model.fit_generator(
    trainGen.generator(),
    steps_per_epoch = trainGen.numImages // 64,
    validation_data = valGen.generator(),
    validation_steps = valGen.numImages // 64,
    epochs = NUM_EPOCHS,
    max_queue_size = 10,
    callbacks = callbacks,
    verbose = 1
)

# close the databases
trainGen.close()
valGen.close()

# save the network to disk
print("[INFO] serializing network...")
model.save(args["model"])
