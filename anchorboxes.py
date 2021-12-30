import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.engine.topology import InputSpec
from keras.engine.topology import Layer
from utils import convert_coordinates


'''
This class defines a keras layers for generating anchor boxes for the grid cells of a feature map given as input
'''
class AnchorBoxes(Layer):

    def __init__(self, img_size, n_boxes, aspect_ratios, scale, next_scale, step, offset = 0.5, **kwargs):

        # Storing the parameters required for calculating the anchor boxes at each grid cell in a feature map

        self.img_size = img_size
        self.scale = scale
        self.next_scale = next_scale
        self.aspect_ratios = aspect_ratios
        self.step = step
        self.offset = offset
        self.n_boxes = n_boxes

        super(AnchorBoxes, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        super(AnchorBoxes, self).build(input_shape)

    def call(self, x, mask=None):

        # list for storing the height and weight
        wh_list = []

        img_h = img_w = self.img_size

        # storing the current input layer shape in corresponding variables
        batch_size, fmap_h, fmap_w, fmap_ch = x._keras_shape

        # Loop to calculate the height and width of the bounding boxes
        # with respect to the specified aspect_ratios and scales
        for ar in self.aspect_ratios:
            # We create 2 anchor boxes of different scale for aspect ratio = 1
            if (ar == 1):
                box_height = box_width = self.scale * img_h
                wh_list.append((box_width, box_height))
                box_height = box_width = np.sqrt(self.scale * self.next_scale) * img_w
                wh_list.append((box_width, box_height))

            # One box for each other aspect ratio
            else:
                box_height = self.scale * img_h / np.sqrt(ar)
                box_width = self.scale * img_w * np.sqrt(ar)
                wh_list.append((box_width, box_height))

        wh_list = np.array(wh_list)

        ### Logic for generation of anchor boxes

        # Calculation of the centriods of anchor boxes in the feature map
        cx = np.linspace(self.offset * self.step, (self.offset + fmap_h - 1) * self.step, fmap_h)
        cy = np.linspace(self.offset * self.step, (self.offset + fmap_w - 1) * self.step, fmap_w)

        # converting it to 2 dimensions as the feature maps are 2D too.
        cx_grid, cy_grid = np.meshgrid(cx, cy)
        cx_grid = np.expand_dims(cx_grid, -1)
        cy_grid = np.expand_dims(cy_grid, -1)

        # Create a 4D tensor template of shape `(feature_map_height, feature_map_width, n_boxes, 4)`
        # where the last dimension will contain `(cx, cy, w, h)`
        boxes_tensor = np.zeros((fmap_h, fmap_w, self.n_boxes, 4))

        # Assigning the generated centriods and height and width to the tensor
        boxes_tensor[:, :, :, 0] = np.tile(cx_grid, (1, 1, self.n_boxes))
        boxes_tensor[:, :, :, 1] = np.tile(cy_grid, (1, 1, self.n_boxes))
        boxes_tensor[:, :, :, 2] = wh_list[:, 0]
        boxes_tensor[:, :, :, 3] = wh_list[:, 1]

        # Convert to corners
        boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion=1)

        # Normalize box coordinates

        boxes_tensor[:, :, :, [0, 2]] /= img_w
        boxes_tensor[:, :, :, [1, 3]] /= img_h

        # Convert to centroids
        boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion=2)

        variances = [0.1, 0.1, 0.2, 0.2]

        # Storing the variances to be appended to the boxes tensor
        variances_tensor = np.zeros_like(boxes_tensor) # Has shape `(feature_map_height, feature_map_width, n_boxes, 4)`
        variances_tensor += variances

        # Now `boxes_tensor` becomes a tensor of shape `(feature_map_height, feature_map_width, n_boxes, 8)`
        boxes_tensor = np.concatenate((boxes_tensor, variances_tensor), axis=-1)


        boxes_tensor = np.expand_dims(boxes_tensor, axis=0)

        boxes_tensor = K.tile(K.constant(boxes_tensor, dtype='float32'),(K.shape(x)[0], 1, 1, 1, 1))

        return boxes_tensor

    def compute_output_shape(self, input_shape):

        batch_size, feature_map_height, feature_map_width, feature_map_channels = input_shape
        return (batch_size, feature_map_height, feature_map_width, self.n_boxes, 8)

    def get_config(self):
        config = {
            'image_size': self.img_size,
            'this_scale': self.scale,
            'next_scale': self.next_scale,
            'aspect_ratios': list(self.aspect_ratios)
        }
        base_config = super(AnchorBoxes, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
