from __future__ import division
import numpy as np
import inspect
from collections import defaultdict
import warnings
from bs4 import BeautifulSoup
import pickle
from copy import deepcopy
from PIL import Image
import cv2
import os
import sys
import pdb
from box_validation_utils import BoxFilter
import sklearn
from tqdm import tqdm

'''

A class that deals with the reading and parsing of the data

'''
class DataGenerator:


    def __init__(self,
                 out_height=300,  # The image size that the read image should be resized to if required
                 out_width=300,   ####
                 filenames=None,  # The list of names of the image files in dataset
                 images_dir=None, # path to directory where images are stored
                 labels=None,     # the ground truth values
                 image_ids=None,  # list of ids of the images
                 eval_neutral=None, # list keeping the record of whether image is diffcult to detect
                 labels_output_format=('class_id', 'xmin', 'ymin', 'xmax', 'ymax'), # format in which the ground truth values are stored
                 verbose=True):   # for internal use to show processing output on the console

        # Initializing class parameters with values passed in the constructor argument

        self.labels_output_format = labels_output_format
        self.labels_format={'class_id': labels_output_format.index('class_id'),
                            'xmin': labels_output_format.index('xmin'),
                            'ymin': labels_output_format.index('ymin'),
                            'xmax': labels_output_format.index('xmax'),
                            'ymax': labels_output_format.index('ymax')}

        self.dataset_size = 0 # As long as we haven't loaded anything yet, the dataset size is zero.
        self.images = []      # Currently will be empty and will be updated batch-wise by the generate function.
        self.filenames = filenames
        self.out_height = out_height
        self.out_width = out_width
        self.labels = labels
        self.img_sizes = [] # Will store the original size of images in order.

        self.transforms_list = [] # list transformations like resize and normalization to be performed on images

    '''
    This function parses the imageset files containing list of name of images in the dataset.
    Then for each name in the list, it reads the annotation xml file and parses the data into ground_truth values format
    and then stored them in varaible labels.
    It is also responsible for updating class variables like self.img_sizes, self.filenames, self.image_ids, etc.
    '''
    def parse_xml(self,
                  images_dirs,      # list of directories containing the image files for the dataset
                  image_set_filenames,  # file contatining list of image file names in the dataset
                  annotations_dirs=[],  # list of directories containing the annotations or ground_truth files for the dataset
                  classes=['background',                                # classes in PASCAL VOC dataset
                           'aeroplane', 'bicycle', 'bird', 'boat',
                           'bottle', 'bus', 'car', 'cat',
                           'chair', 'cow', 'diningtable', 'dog',
                           'horse', 'motorbike', 'person', 'pottedplant',
                           'sheep', 'sofa', 'train', 'tvmonitor'],
                  include_classes = 'all'):                             # Specify whether all classes are included

            # Set class members.
            self.images_dirs = images_dirs
            self.annotations_dirs = annotations_dirs
            self.image_set_filenames = image_set_filenames
            self.classes = classes
            self.include_classes = include_classes

            # Erase data that might have been parsed before.
            self.filenames = []
            self.image_ids = []
            self.labels = []
            self.eval_neutral = []
            print("Parsing data ...")

            # Loop to read and parse data
            for images_dir, image_set_filename, annotations_dir in zip(images_dirs, image_set_filenames, annotations_dirs):

                # Read the image set file that to update IDs of the images in the dataset.This is important because
                # images and their ground_truth files have the same ID.
                with open(image_set_filename) as f:
                    image_ids = [line.strip() for line in f]
                    self.image_ids += image_ids

                # This will show a progress bar on the output screen
                it = tqdm(image_ids, desc="Reading images from set: '{}'".format(os.path.basename(image_set_filename)), file=sys.stdout)

                # Loop over all images in this dataset while updating the progress bar created in the above step.
                for image_id in it:

                    # Update the list filenames with the path to the image.
                    filename = '{}'.format(image_id) + '.jpg'
                    self.filenames.append(os.path.join(images_dir, filename))

                    # Storing labels and image sizes from the xml file in the class variables
                    if not annotations_dir is None:
                        # Parse the XML file for this image.
                        with open(os.path.join(annotations_dir, image_id + '.xml')) as f:
                            soup = BeautifulSoup(f, 'xml')

                        folder = soup.folder.text
                        filename = soup.filename.text

                        boxes = [] # We'll store all bounding boxes for this image here.
                        eval_neutr = []
                        objects = soup.find_all('object') # Get a list of all objects in this image.

                        # store the size of the current image
                        size = (int(soup.find('size').height.text), int(soup.find('size').width.text))

                        # Parse the data for each object.
                        for obj in objects:
                            class_name = obj.find('name', recursive=False).text
                            class_id = self.classes.index(class_name)

                            difficult = int(obj.find('difficult', recursive=False).text)
                            # Get the bounding box coordinates.
                            bndbox = obj.find('bndbox', recursive=False)
                            xmin = int(bndbox.xmin.text)
                            ymin = int(bndbox.ymin.text)
                            xmax = int(bndbox.xmax.text)
                            ymax = int(bndbox.ymax.text)

                            # dictionary of items read from the xml file
                            item_dict = {'folder': folder,
                                         'image_name': filename,
                                         'image_id': image_id,
                                         'class_name': class_name,
                                         'class_id': class_id,
                                         'xmin': xmin,
                                         'ymin': ymin,
                                         'xmax': xmax,
                                         'ymax': ymax}
                            box = []

                            # Storing the values in a list in the specified format
                            for item in self.labels_output_format:
                                box.append(item_dict[item])

                            # Storing the read labels list as a list element in another list
                            boxes.append(box)
                            if difficult: eval_neutr.append(True)
                            else: eval_neutr.append(False)

                        # Storing the list of labels per object as a element of a list.
                        # This means, the self.labels will be list of labels where
                        # each element contains a list of labels for all objects found in an image.
                        # Basically a list of lists.

                        self.labels.append(boxes)
                        self.eval_neutral.append(eval_neutr)
                        # update the variable with the image sizes.
                        self.img_sizes.append(size)

            self.dataset_size = len(self.filenames)
            self.dataset_indices = np.arange(self.dataset_size, dtype=np.int32)

            print("Parsing data complete")

    '''
    The function responsible to yield data specified in the argument returns in batches of specified size.
    The returns include:

        1. 'processed_images' - images processed according to the model requirements.
        2. 'encoded_labels'   - ground truth values encoded in a format used by the loss function of the model.
        3. 'image_ids'        - list of IDs of the images
        4. 'original_images'  - the original images without any processing
        5. 'original_labels'  - non-encoded labels, in the form in which they were read
        6. 'image_sizes'      - list of image sizes corresponding to the current batch

    This output is fed into keras' fit_generator function.

    '''
    def generate(self,
         batch_size=32,     # Batch size to yield at each iteration
         shuffle=True,      # Shuffle the data for better training
         transformations=[],    # This is mainly for data_augmentation
         label_encoder=None,    # The encoder object that will encode the ground_truth values in required form
         returns={'processed_images', 'encoded_labels'}, # Specify what to return
         keep_images_without_gt=False):  # Specify whether to keep images without ground_truth values

        # Initializing class params
        self.transforms_list = transformations

        # Shuffle the data for better training
        if shuffle:

            objects_to_shuffle = [self.dataset_indices]
            if not (self.filenames is None):
                objects_to_shuffle.append(self.filenames)
            if not (self.labels is None):
                objects_to_shuffle.append(self.labels)
            if not (self.image_ids is None):
                objects_to_shuffle.append(self.image_ids)
            if not (self.eval_neutral is None):
                objects_to_shuffle.append(self.eval_neutral)

            shuffled_objects = sklearn.utils.shuffle(*objects_to_shuffle)
            for i in range(len(objects_to_shuffle)):
                objects_to_shuffle[i][:] = shuffled_objects[i]

        # Instantiating object from BoxFilter class that checks for degenerate boxes(i.e. boxes that are not possible).
        # for eg. box having 'xmin' greater than 'xmax' etc.
        box_filter = BoxFilter(check_overlap=False,
                               check_min_area=False,
                               check_degenerate=True,
                               labels_format=self.labels_format)

        # Override the labels formats of all the transformations to make sure they are set correctly.
        if not (self.labels is None):
            for transform in transformations:
                transform.labels_format = self.labels_format

        #### Loop to store and update values to return for each batch

        current = 0 # stores the initial index of the current batch
        while True:

            # These will store the images and labels for the current batch
            batch_x, batch_y = [], []

            # If the fit_generator steps are still left and current has exceeded dataset size limit
            # start over from the first batch of dataset
            if current>=self.dataset_size:
                current=0

            if shuffle:
                objects_to_shuffle = [self.dataset_indices]
                if not (self.filenames is None):
                    objects_to_shuffle.append(self.filenames)
                if not (self.labels is None):
                    objects_to_shuffle.append(self.labels)
                if not (self.image_ids is None):
                    objects_to_shuffle.append(self.image_ids)
                if not (self.eval_neutral is None):
                    objects_to_shuffle.append(self.eval_neutral)
                shuffled_objects = sklearn.utils.shuffle(*objects_to_shuffle)
                for i in range(len(objects_to_shuffle)):
                    objects_to_shuffle[i][:] = shuffled_objects[i]

            batch_original_images = []

            it = self.filenames[current:current+batch_size]

            # Reading and storing images batch-wise
            for filename in it:
                with Image.open(filename) as image:
                    batch_x.append(np.array(image, dtype = np.uint8))

            # storing the values of labels for current batch in 'batch_y'
            batch_y = deepcopy(self.labels[current:current+batch_size])

            # store original images and labels before any processing done on them
            # if they are to be returned

            if 'original_images' in returns:
                batch_original_images = deepcopy(batch_x)
            if 'original_labels' in returns:
                batch_original_labels = deepcopy(batch_y) # The original, unaltered labels

            if not (self.eval_neutral is None):
                batch_eval_neutral = self.eval_neutral[current:current+batch_size]
            else:
                batch_eval_neutral = None
            batch_image_ids = self.image_ids[current:current+batch_size]
            img_sizes = self.img_sizes[current:current+batch_size]

            # increment current for the next batch
            current += batch_size

            #### Resizing the images and labels

            batch_items_to_remove = [] # In case we need to remove any images from the batch, store their indices in this list.
            batch_inverse_transforms = []

            # Loop for pre-processing of the images and labels
            for i in range(len(batch_x)):

                if not (self.labels is None):
                    # Convert the labels for this image to an array (in case they aren't already).
                    batch_y[i] = np.array(batch_y[i])

                    # If this image has no ground truth boxes, maybe we don't want to keep it in the batch.

                    if (batch_y[i].size == 0):
                        batch_items_to_remove.append(i)
                        batch_inverse_transforms.append([])
                        continue

                if self.transforms_list:

                    # Transformations are applied to the data
                    for transform in self.transforms_list:

                        if not (self.labels is None):
                            batch_x[i], batch_y[i] = transform(batch_x[i], batch_y[i])

                        else:
                            batch_x[i] = transform(batch_x[i])

                else:
                    batch_x[i], batch_y[i] = self.transform(batch_x[i], batch_y[i])

                if batch_x[i] is None: # In case the transform failed to produce an output image, which is possible for some random transforms.
                    batch_items_to_remove.append(i)
                    continue

                if not (self.labels is None):

                    xmin = self.labels_format['xmin']
                    ymin = self.labels_format['ymin']
                    xmax = self.labels_format['xmax']
                    ymax = self.labels_format['ymax']

                    batch_y[i] = box_filter(batch_y[i])
                    if (batch_y[i].size == 0) and not keep_images_without_gt:
                        batch_items_to_remove.append(i)

            # If there are some data items that are not fit for training, remove them
            if batch_items_to_remove:
                for j in sorted(batch_items_to_remove, reverse=True):
                    # This isn't efficient, but it hopefully shouldn't need to be done often anyway.
                    batch_x.pop(j)
                    batch_filenames.pop(j)
                    if batch_inverse_transforms: batch_inverse_transforms.pop(j)
                    if not (self.labels is None): batch_y.pop(j)
                    if not (self.image_ids is None): batch_image_ids.pop(j)
                    if not (self.eval_neutral is None): batch_eval_neutral.pop(j)
                    if 'original_images' in returns: batch_original_images.pop(j)
                    if 'original_labels' in returns and not (self.labels is None): batch_original_labels.pop(j)

            # converting the list of images into ndarray of images for easy processing and input to CNN
            batch_x = np.array(batch_x)

            # Encode the ground_truth in a format taken by the CNN for training
            if 'encoded_labels' in returns:
                batch_y_encoded = label_encoder(batch_y)

            ret = []

            if 'processed_images' in returns: ret.append(batch_x)
            if 'encoded_labels' in returns: ret.append(batch_y_encoded)
            if 'image_ids' in returns: ret.append(batch_image_ids)
            if 'evaluation-neutral' in returns: ret.append(batch_eval_neutral)
            if 'original_images' in returns: ret.append(batch_original_images)
            if 'original_labels' in returns: ret.append(batch_original_labels)
            if 'image_sizes' in returns: ret.append(img_sizes)
            yield ret

    def get_dataset_size(self):
        '''
        Returns:
            The number of images in the dataset.
        '''
        return self.dataset_size

    '''
    This function resizes the images and labels into size accepted by the CNN
    '''
    def transform(self, image, labels=None):

        # Initialize necessary variables
        img_height, img_width = image.shape[:2]

        # Set the format (indices for elements in labels)
        xmin = self.labels_format['xmin']
        ymin = self.labels_format['ymin']
        xmax = self.labels_format['xmax']
        ymax = self.labels_format['ymax']

        # select an interpolation mode for resizing
        interpolation_mode = np.random.choice([cv2.INTER_NEAREST,
                                              cv2.INTER_LINEAR,
                                              cv2.INTER_CUBIC,
                                              cv2.INTER_AREA,
                                              cv2.INTER_LANCZOS4])

        # Resize the image with specified dimensions
        image = cv2.resize(image, dsize=(self.out_width, self.out_height), interpolation=interpolation_mode)

        # Update the bounding box coordinates to correspond with the resized image
        if not (labels is None):

            labels = np.copy(labels)
            labels[:, [ymin, ymax]] = np.round(labels[:, [ymin, ymax]] * (self.out_height / img_height), decimals=0)
            labels[:, [xmin, xmax]] = np.round(labels[:, [xmin, xmax]] * (self.out_width / img_width), decimals=0)

            # Check for degenerate boxes
            requirements_met = np.ones(shape=labels.shape[0], dtype=np.bool)

            non_degenerate = (labels[:,xmax] > labels[:,xmin]) * (labels[:,ymax] > labels[:,ymin])
            requirements_met *= non_degenerate

            # Store valid bounding boxes
            valid_labels = labels[requirements_met]

            return image, valid_labels

        else:
            return image
