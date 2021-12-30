from __future__ import division
import numpy as np
from keras.models import Model
from keras.layers import Input, Lambda, Activation, Conv2D, MaxPooling2D, ZeroPadding2D, Reshape, Concatenate
from keras.regularizers import l2
import keras.backend as K

from anchorboxes import AnchorBoxes
from normalize import L2Normalization

'''

This method creates and returns the CNN model of SSD 300 as described in the paper

'''
def original_ssd(image_size=[300,300,3], # Input image shape
            num_classes=20, # Classes to be predicted
            l2_regularization=0.0005, # Regularization Value as used in the paper
            scales=[1.0, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05], # Scales for PASCAL VOC 2007+12 datasets used in the paper
            aspect_ratios=[[1.0, 2.0, 0.5],                 ############################################
                           [1.0, 2.0, 0.5, 3.0, 1.0/3.0],   #
                           [1.0, 2.0, 0.5, 3.0, 1.0/3.0],   # Aspect ratios for generating anchor boxes
                           [1.0, 2.0, 0.5, 3.0, 1.0/3.0],   # for each feature map layer respectively
                           [1.0, 2.0, 0.5],                 #
                           [1.0, 2.0, 0.5]],                ############################################
            steps=[8, 16, 32, 64, 100, 300], # Difference between the center pixels of any two anchor boxes for each feature map layer respectively
            variances=[0.1, 0.1, 0.2, 0.2],  # to account for variances in the predictions
            return_predictor_sizes=False): # Predictor layer sizes are returned if true

    # Declaring some variables needed to construct the model

    n_predictor_layers = 6 # number of layers that will act as feature maps. (i.e. will predict the bounding boxes and confidence values)

    num_classes += 1 # Number of classes to predict (We take in more class in addition to account for the background class)

    l2_reg = l2_regularization # Make the internal name shorter.

    img_height, img_width, img_channels = image_size[0], image_size[1], image_size[2] # Image sizes to be given as input

    num_boxes = [4, 6, 6, 6, 4, 4] # The number of anchor boxes per grid cell in each of the predictor layers respectively

    ############################################################################
    # Build the network.
    ############################################################################

    input_shape = Input(shape=(img_height, img_width, img_channels))

    #### Base network

    # Layer 1
    conv1_1 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv1_1')(input_shape)
    conv1_2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv1_2')(conv1_1)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool1')(conv1_2)

    # Layer 2
    conv2_1 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv2_1')(pool1)
    conv2_2 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv2_2')(conv2_1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool2')(conv2_2)

    # Layer 3
    conv3_1 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv3_1')(pool2)
    conv3_2 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv3_2')(conv3_1)
    conv3_3 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv3_3')(conv3_2)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool3')(conv3_3)

    # Layer 4
    conv4_1 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv4_1')(pool3)
    conv4_2 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv4_2')(conv4_1)
    conv4_3 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv4_3')(conv4_2)
    pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool4')(conv4_3)

    # Layer 5
    conv5_1 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv5_1')(pool4)
    conv5_2 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv5_2')(conv5_1)
    conv5_3 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv5_3')(conv5_2)
    pool5 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='pool5')(conv5_3)

    #### End of the base network

    # Dense layers Converted into convolutional layers for efficient computation
    # These also act as links between the base netwrok and the predictor network
    fc6 = Conv2D(1024, (3, 3), dilation_rate=(6, 6), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='fc6')(pool5)

    fc7 = Conv2D(1024, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='fc7')(fc6)

    ### Predictor network

    # Feature map layers
    conv8_1 = Conv2D(256, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv8_1')(fc7)
    conv8_2 = Conv2D(512, (3, 3), strides=(2, 2), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv8_2')(conv8_1)

    conv9_1 = Conv2D(128, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv9_1')(conv8_2)
    conv9_2 = Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv9_2')(conv9_1)

    conv10_1 = Conv2D(128, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv10_1')(conv9_2)
    conv10_2 = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv10_2')(conv10_1)

    conv11_1 = Conv2D(128, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv11_1')(conv10_2)
    conv11_2 = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv11_2')(conv11_1)

    # Normalizing the first feature map
    conv4_3_norm = L2Normalization(gamma_init=20, name='conv4_3_norm')(conv4_3)

    # Confidence layers - will output the class confidence values for each prior box hence the output shape of each layer is:
    # (batch_size, height, width, num_boxes * num_classes)
    conv4_3_norm_conf = Conv2D(num_boxes[0] * num_classes, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conf1')(conv4_3_norm)
    fc7_conf = Conv2D(num_boxes[1] * num_classes, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conf2')(fc7)
    conv8_2_conf = Conv2D(num_boxes[2] * num_classes, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conf3')(conv8_2)
    conv9_2_conf = Conv2D(num_boxes[3] * num_classes, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conf4')(conv9_2)
    conv10_2_conf = Conv2D(num_boxes[4] * num_classes, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conf5')(conv10_2)
    conv11_2_conf = Conv2D(num_boxes[5] * num_classes, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conf6')(conv11_2)

    # Localization layers - will output 4 offset values of box coordinates and hence the output shape is:
    # (batch, height, width, num_boxes * 4)
    conv4_3_norm_loc = Conv2D(num_boxes[0] * 4, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='loc1')(conv4_3_norm)
    fc7_loc = Conv2D(num_boxes[1] * 4, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='loc2')(fc7)
    conv8_2_loc = Conv2D(num_boxes[2] * 4, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='loc3')(conv8_2)
    conv9_2_loc = Conv2D(num_boxes[3] * 4, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='loc4')(conv9_2)
    conv10_2_loc = Conv2D(num_boxes[4] * 4, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='loc5')(conv10_2)
    conv11_2_loc = Conv2D(num_boxes[5] * 4, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='loc6')(conv11_2)

    ### Generate the anchor boxes from each feature map layer
    # Output shape of anchors: (batch, height, width, num_boxes, 8)
    # The last shape consists of 4 box coordinates and 4 variance values and hence 8 values per box.

    conv4_3_norm_priorbox = AnchorBoxes(300, num_boxes[0], aspect_ratios[0], scales[0], scales[1], steps[0])(conv4_3_norm_loc)
    fc7_priorbox = AnchorBoxes(300, num_boxes[1], aspect_ratios[1], scales[1], scales[2], steps[1])(fc7_loc)
    conv8_2_priorbox = AnchorBoxes(300, num_boxes[2], aspect_ratios[2], scales[2], scales[3], steps[2])(conv8_2_loc)
    conv9_2_priorbox = AnchorBoxes(300, num_boxes[3], aspect_ratios[3], scales[3], scales[4], steps[3])(conv9_2_loc)
    conv10_2_priorbox = AnchorBoxes(300, num_boxes[4], aspect_ratios[4], scales[4], scales[5], steps[4])(conv10_2_loc)
    conv11_2_priorbox = AnchorBoxes(300, num_boxes[5], aspect_ratios[5], scales[5], scales[6], steps[5])(conv11_2_loc)

    ### Reshape layers
    # Reshaping the confidence layers, localization and anchor box layers to a shape on which loss function can operate
    # The classes are isolated in the last axis to perform softmax on them

    # Reshaping confidence layers to shape - (batch, height * width * num_boxes, num_classes)
    conv4_3_norm_conf_reshape = Reshape((-1, num_classes), name='conv4_3_norm_conf_reshape')(conv4_3_norm_conf)
    fc7_conf_reshape = Reshape((-1, num_classes), name='fc7_conf_reshape')(fc7_conf)
    conv8_2_conf_reshape = Reshape((-1, num_classes), name='conv8_2_conf_reshape')(conv8_2_conf)
    conv9_2_conf_reshape = Reshape((-1, num_classes), name='conv9_2_conf_reshape')(conv9_2_conf)
    conv10_2_conf_reshape = Reshape((-1, num_classes), name='conv10_2_conf_reshape')(conv10_2_conf)
    conv11_2_conf_reshape = Reshape((-1, num_classes), name='conv11_2_conf_reshape')(conv11_2_conf)

    # Reshaping localization layers to shape: (batch, height * width * num_boxes, 4)
    # The four box coordinates are isolated in the last axis to compute the smooth L1 loss

    conv4_3_norm_loc_reshape = Reshape((-1, 4), name='conv4_3_norm_loc_reshape')(conv4_3_norm_loc)
    fc7_loc_reshape = Reshape((-1, 4), name='fc7_loc_reshape')(fc7_loc)
    conv8_2_loc_reshape = Reshape((-1, 4), name='conv8_2_loc_reshape')(conv8_2_loc)
    conv9_2_loc_reshape = Reshape((-1, 4), name='conv9_2_loc_reshape')(conv9_2_loc)
    conv10_2_loc_reshape = Reshape((-1, 4), name='conv10_2_loc_reshape')(conv10_2_loc)
    conv11_2_loc_reshape = Reshape((-1, 4), name='conv11_2_loc_reshape')(conv11_2_loc)

    # Reshaping the anchor box layers to shape - `(batch, height * width * num_boxes, 8)`

    conv4_3_norm_priorbox_reshape = Reshape((-1, 8), name='conv4_3_norm_priorbox_reshape')(conv4_3_norm_priorbox)
    fc7_priorbox_reshape = Reshape((-1, 8), name='fc7_priorbox_reshape')(fc7_priorbox)
    conv8_2_priorbox_reshape = Reshape((-1, 8), name='conv8_2_priorbox_reshape')(conv8_2_priorbox)
    conv9_2_priorbox_reshape = Reshape((-1, 8), name='conv9_2_priorbox_reshape')(conv9_2_priorbox)
    conv10_2_priorbox_reshape = Reshape((-1, 8), name='conv10_2_priorbox_reshape')(conv10_2_priorbox)
    conv11_2_priorbox_reshape = Reshape((-1, 8), name='conv11_2_priorbox_reshape')(conv11_2_priorbox)

    ### Concatenating the output from different prediction layers into one layer to perform activation operation and shape the final output

    # Output shape of `conf`: (batch, num_boxes_total, num_classes)
    conf = Concatenate(axis=1, name='conf')([conv4_3_norm_conf_reshape,
                                                       fc7_conf_reshape,
                                                       conv8_2_conf_reshape,
                                                       conv9_2_conf_reshape,
                                                       conv10_2_conf_reshape,
                                                       conv11_2_conf_reshape])

    # Output shape of `loc`: (batch, num_boxes_total, 4)
    loc = Concatenate(axis=1, name='loc')([conv4_3_norm_loc_reshape,
                                                     fc7_loc_reshape,
                                                     conv8_2_loc_reshape,
                                                     conv9_2_loc_reshape,
                                                     conv10_2_loc_reshape,
                                                     conv11_2_loc_reshape])

    # Output shape of `priorbox`: (batch, num_boxes_total, 8)
    priorbox = Concatenate(axis=1, name='priorbox')([conv4_3_norm_priorbox_reshape,
                                                               fc7_priorbox_reshape,
                                                               conv8_2_priorbox_reshape,
                                                               conv9_2_priorbox_reshape,
                                                               conv10_2_priorbox_reshape,
                                                               conv11_2_priorbox_reshape])

    # The box coordinate predictions will go into the loss function just the way they are,
    # but for the class predictions, we'll apply a softmax activation layer first
    conf_softmax = Activation('softmax', name='conf_softmax')(conf)

    # Concatenate the class and box predictions and the anchors to one large predictions vector
    # Output shape of `predictions`: (batch, num_boxes_total, num_classes + 4 + 8)
    predictions = Concatenate(axis=2, name='predictions')([conf_softmax, loc, priorbox])

    # Creating a keras model object
    model = Model(inputs=input_shape, outputs=predictions)

    # The shapes of predictor layers are needed to encode the labels in similar fashion for the loss function to operate on
    if return_predictor_sizes:
        predictor_sizes = np.array([conv4_3_norm_conf._keras_shape[1:3],
                                     fc7_conf._keras_shape[1:3],
                                     conv8_2_conf._keras_shape[1:3],
                                     conv9_2_conf._keras_shape[1:3],
                                     conv10_2_conf._keras_shape[1:3],
                                     conv11_2_conf._keras_shape[1:3]])
        return model, predictor_sizes
    else:
        return model
