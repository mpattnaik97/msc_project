from __future__ import division
import numpy as np
from anchorboxes import AnchorBoxes
import keras.backend as K
from losses import SSDLoss
from ssd_v2 import another_ssd
from data_format import DataGenerator
from encode_input import SSDInputEncoder
import keras
from math import ceil
# Initializing required parameters

img_height = 345 # Height of the model input images
img_width = 345 # Width of the model input images

img_channels = 3 # Number of color channels of the model input images
normalize_coords = True

K.clear_session() # Clear previous models from memory.

model, predictor_sizes = another_ssd(return_predictor_sizes = True, l2_regularization=0.0005) # This creates the scaled ssd model

# Initializing an object of DataGenerator class for reading, parsing and generating the data in batches
train_dataset = DataGenerator(out_height=img_height, out_width=img_width)
val_dataset = DataGenerator(out_height=img_height, out_width=img_width)

# Storing the path for necessary files and directories needed for data generation

# The directories that contain the images.
VOC_2007_images_dir      = '../VOCdevkit/VOC2007/JPEGImages/'
VOC_2012_images_dir      = '../VOCdevkit/VOC2012/JPEGImages/'

# The directories that contain the annotations.
VOC_2007_annotations_dir      = '../VOCdevkit/VOC2007/Annotations/'
VOC_2012_annotations_dir      = '../VOCdevkit/VOC2012/Annotations/'

# The paths to the image sets.
VOC_2007_trainval_image_set_filename = '../VOCdevkit/VOC2007/ImageSets/Main/trainval.txt'
VOC_2012_trainval_image_set_filename = '../VOCdevkit/VOC2012/ImageSets/Main/trainval.txt'
VOC_2007_test_image_set_filename     = '../VOCdevkit/VOC2007/ImageSets/Main/test.txt'

# List of class names where the indices are their respective class IDs
classes=[
        'background', 'aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike',
        'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]

print("Parsing data ... ")

# Read, parse and store image information and ground_truth values in class variables

train_dataset.parse_xml(images_dirs=[VOC_2007_images_dir, VOC_2012_images_dir],
                        image_set_filenames=[VOC_2007_trainval_image_set_filename, VOC_2012_trainval_image_set_filename],
                        annotations_dirs=[VOC_2007_annotations_dir, VOC_2012_annotations_dir],
                        classes=classes,
                        include_classes='all')

val_dataset.parse_xml(images_dirs=[VOC_2007_images_dir],
                      image_set_filenames=[VOC_2007_test_image_set_filename],
                      annotations_dirs=[VOC_2007_annotations_dir],
                      classes=classes,
                      include_classes='all')

print("Parsing complete")

# Set the batch size, normally 32 but can be decreased if need be
batch_size = 10

# Instantiate input encoder class object for encoding the input labels
ssd_input_encoder = SSDInputEncoder(img_height=img_height,
                                    img_width=img_width,
                                    predictor_sizes=predictor_sizes,
                                    steps = [9, 18, 36, 73, 115, 345])

# Create a generator object that will generate the training data in batches
train_generator = train_dataset.generate(batch_size=batch_size,
                                         label_encoder=ssd_input_encoder,
                                         returns={'processed_images',
                                                  'encoded_labels'},
                                         keep_images_without_gt=False)

# Create a generator object that will generate the validation data in batches
val_generator = val_dataset.generate(batch_size=batch_size,
                                     label_encoder=ssd_input_encoder,
                                     returns={'processed_images',
                                              'encoded_labels'},
                                     keep_images_without_gt=False)

train_dataset_size = train_dataset.get_dataset_size()
val_dataset_size   = val_dataset.get_dataset_size()

print("Number of images in the training dataset:\t{:>6}".format(train_dataset_size))
print("Number of images in the validation dataset:\t{:>6}".format(val_dataset_size))

# uncomment if you want to load a model with specific weights and give the file path to the model
# path = ''
# model.load_weights(path)

# declaring parameters for training
adam = keras.optimizers.Adam(lr=0.001) # optimizer

ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0) # loss

model.compile(optimizer=adam, loss=ssd_loss.compute_loss) #compiling the model

# schedule learning rate
def lr_schedule(epoch):

    if epoch < 30:
        return 0.001
    elif epoch < 70:
        return 0.0001
    else:
        return 0.00001

# Defining the callback functions for keras model

# updates the learning rate
learning_rate_scheduler = keras.callbacks.LearningRateScheduler(schedule=lr_schedule, verbose=1)

# terminates training when loss becomes NaN
terminate_on_nan = keras.callbacks.TerminateOnNaN()

# logs the results of training in the file to which the path is provided as argument
log_results = keras.callbacks.CSVLogger('../results/another_ssd.log', append=True)

# saves the model weights at every 20 epochs into a new file
checkpoint = keras.callbacks.ModelCheckpoint(filepath='../saved_models/another_ssd_weights_{epoch:02d}.hdf5', save_weights_only=True, period = 20)

# list of callback functions
callbacks = [learning_rate_scheduler, terminate_on_nan, log_results, checkpoint]

# necessary params
initial_epoch   = 0
final_epoch     = 80
steps_per_epoch = 1000

# calling keras API function fit_generator to train the model in batches for given epochs
history = model.fit_generator(generator=train_generator,
                              validation_data=val_generator,
                              validation_steps=ceil(val_dataset_size/batch_size),
                              steps_per_epoch=steps_per_epoch,
                              epochs=final_epoch,
                              callbacks=callbacks,
                              initial_epoch=initial_epoch)
