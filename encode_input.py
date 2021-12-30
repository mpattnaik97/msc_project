from __future__ import division
from utils import convert_coordinates
import numpy as np
import pdb

from utils import iou, match_bipartite_greedy, match_multi

'''
This class is responsible for converting the given input labels into encoded form
such that it can be used by the CNN loss function to train the model.
'''
class SSDInputEncoder:

    def __init__(self,
                 img_height,
                 img_width,
                 predictor_sizes,
                 n_classes = 20,
                 scales = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05],     # Scales for PASCAL VOC 2007+12 datasets used in the paper
                 aspect_ratios = [[1.0, 2.0, 0.5],                      ############################################
                                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],         #
                                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],         # Aspect ratios for generating anchor boxes
                                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],         # for each feature map layer respectively
                                 [1.0, 2.0, 0.5],                       #
                                 [1.0, 2.0, 0.5]],                      ############################################
                 steps=[8, 16, 32, 64, 100, 300],          # Difference between the center pixels of any two anchor boxes for each feature map layer respectively
                 variances=[0.1, 0.1, 0.2, 0.2],           # to account for variances in the predictions
                 offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],   # It is the space between to center points of anchorboxes in adjacent gridcells in one feature map
                 pos_iou_threshold=0.5,    # Threshold value for iou of a predicted bounding box to be counted as true positive
                 neg_iou_limit=0.3,        # Threshold value for iou of a predicted bounding box below which any bounding box to be counted as false positive
                 border_pixels='half',     # Determine how to include border pixels for a anchor box
                 coords='centroids',       # Format in which the bounding box coordinates should be in
                 normalize_coords=True,    # Normalizes bounding box coordinate values
                 background_id=0):

        # Initializing and setting up required parameters

        predictor_sizes = np.array(predictor_sizes)
        if predictor_sizes.ndim == 1:
            predictor_sizes = np.expand_dims(predictor_sizes, axis=0)


        self.img_height = img_height
        self.img_width = img_width
        self.n_classes = n_classes + 1 # + 1 for the background class
        self.predictor_sizes = predictor_sizes
        self.scales = scales
        self.aspect_ratios = aspect_ratios
        self.steps = steps
        self.offsets = offsets
        self.variances = variances
        self.pos_iou_threshold = pos_iou_threshold
        self.neg_iou_limit = neg_iou_limit
        self.border_pixels = border_pixels
        self.coords = coords
        self.normalize_coords = normalize_coords
        self.background_id = background_id

        self.n_boxes = []
        self.n_boxes.append(len(self.aspect_ratios) + 1)


        # compute anchor boxes

        self.boxes_list = [] # This will store the anchor boxes for each predictor layer.

        # The following lists just store anchor boxes information.

        self.wh_list_diag = [] # Box widths and heights for each predictor layer
        self.steps_diag = [] # Horizontal and vertical distances between any two boxes for each predictor layer
        self.offsets_diag = [] # Offsets for each predictor layer
        self.centers_diag = [] # Anchor box center points as `(cy, cx)` for each predictor layer

        # Iterate over all feature maps or predictor layers and compute the anchor boxes for each one.

        for i in range(len(self.predictor_sizes)):
            boxes, center, wh, step, offset = self.generate_anchor_boxes_for_layer(feature_map_size=self.predictor_sizes[i],
                                                                                   aspect_ratios=self.aspect_ratios[i],
                                                                                   this_scale=self.scales[i],
                                                                                   next_scale=self.scales[i+1],
                                                                                   this_steps=self.steps[i],
                                                                                   this_offsets=self.offsets[i],
                                                                                   diagnostics=True)

            self.boxes_list.append(boxes)
            self.wh_list_diag.append(wh)
            self.steps_diag.append(step)
            self.offsets_diag.append(offset)
            self.centers_diag.append(center)

    def __call__(self, ground_truth_labels):

        # Setting the indices as names of relevent label field
        class_id = 0
        xmin = 1
        ymin = 2
        xmax = 3
        ymax = 4

        # Getting the batch size
        batch_size = len(ground_truth_labels)

        # Initializing the variable that will store the encoded labels
        y_encoded = self.generate_encoding_template(batch_size=batch_size, diagnostics=False)

        y_encoded[:, :, self.background_id] = 1 # All boxes are background boxes by default.
        n_boxes = y_encoded.shape[1] # The total number of boxes that the model predicts per batch item
        class_vectors = np.eye(self.n_classes) # An identity matrix that we'll use as one-hot class vectors

        for i in range(batch_size): # For each batch item...

            # If the current item does not have a ground truth skip the below steps and start over with the next item
            if ground_truth_labels[i].size == 0: continue

            # converting the ground truth labels to float for processing
            labels = ground_truth_labels[i].astype(np.float)

            # normalize the box coordinates.
            if self.normalize_coords:
                labels[:,[ymin,ymax]] /= self.img_height # Normalize ymin and ymax relative to the image height
                labels[:,[xmin,xmax]] /= self.img_width # Normalize xmin and xmax relative to the image width

            # Convert the box coordinates from corners to centriods (i.e. from (xmin, ymin, xmax, ymax) to (cx, cy, w, h))
            labels = convert_coordinates(labels, start_index=1, conversion=2)

            classes_one_hot = class_vectors[labels[:, class_id].astype(np.int)] # The one-hot class IDs for the ground truth boxes of this batch item
            labels_one_hot = np.concatenate([classes_one_hot, labels[:, [xmin,ymin,xmax,ymax]]], axis=-1) # The one-hot version of the labels for this batch item



            # Compute the IoU similarities between all anchor boxes and all ground truth boxes for this batch item.
            # This is a matrix of shape `(num_ground_truth_boxes, num_anchor_boxes)`.
            similarities = iou(labels[:,[xmin,ymin,xmax,ymax]], y_encoded[i,:,-12:-8], coords=self.coords, mode='outer_product', border_pixels=self.border_pixels)

            # Perform bipartite matching, i.e. match each ground truth box to the one anchor box with the highest IoU.
            # This ensures that each ground truth box will have at least one good match.

            # For each ground truth box, get the anchor box to match with it.
            bipartite_matches = match_bipartite_greedy(weight_matrix=similarities)

            # Write the ground truth data to the matched anchor boxes.
            y_encoded[i, bipartite_matches, :-8] = labels_one_hot

            # Set the columns of the matched anchor boxes to zero to indicate that they were matched.
            similarities[:, bipartite_matches] = 0

            #  Perform 'multi' matching, where each remaining anchor box will be matched to its most similar
            #  ground truth box with an IoU of at least `pos_iou_threshold`, or not matched if there is no
            #  such ground truth box.

            # Get all matches that satisfy the IoU threshold.
            matches = match_multi(weight_matrix=similarities, threshold=self.pos_iou_threshold)

            # Write the ground truth data to the matched anchor boxes.
            y_encoded[i, matches[1], :-8] = labels_one_hot[matches[0]]

            # Set the columns of the matched anchor boxes to zero to indicate that they were matched.
            similarities[:, matches[1]] = 0

            # Now after the matching is done, all negative (background) anchor boxes that have
            # an IoU of `neg_iou_limit` or more with any ground truth box will be set to netral,
            # i.e. they will no longer be background boxes. These anchors are "too close" to a
            # ground truth box to be valid background boxes.

            max_background_similarities = np.amax(similarities, axis=0)
            neutral_boxes = np.nonzero(max_background_similarities >= self.neg_iou_limit)[0]
            y_encoded[i, neutral_boxes, self.background_id] = 0

        # Convert box coordinates to anchor box offsets.

        y_encoded[:,:,[-12,-11]] -= y_encoded[:,:,[-8,-7]] # cx(gt) - cx(anchor), cy(gt) - cy(anchor)
        y_encoded[:,:,[-12,-11]] /= y_encoded[:,:,[-6,-5]] * y_encoded[:,:,[-4,-3]] # (cx(gt) - cx(anchor)) / w(anchor) / cx_variance, (cy(gt) - cy(anchor)) / h(anchor) / cy_variance
        y_encoded[:,:,[-10,-9]] /= y_encoded[:,:,[-6,-5]] # w(gt) / w(anchor), h(gt) / h(anchor)
        y_encoded[:,:,[-10,-9]] = np.log(y_encoded[:,:,[-10,-9]]) / y_encoded[:,:,[-2,-1]]

        return y_encoded

    '''
    This function generates 8732 anchor boxes for all the predictor layers (feature maps) and returns them to get stored
    in the encoded labels.
    '''
    def generate_anchor_boxes_for_layer(self,
                                        feature_map_size,
                                        aspect_ratios,
                                        this_scale,
                                        next_scale,
                                        this_steps=None,
                                        this_offsets=None,
                                        diagnostics=False):

        # Compute box width and height for each aspect ratio.

        # The shorter side of the image will be used to compute `w` and `h` using `scale` and `aspect_ratios`.
        size = min(self.img_height, self.img_width)

        # list for storing the height and weight
        wh_list = []

        # Loop to calculate the height and width of the bounding boxes
        # with respect to the specified aspect_ratios and scales

        for ar in aspect_ratios:

            if (ar == 1):

                # We create 2 anchor boxes of different scale for aspect ratio = 1
                box_height = box_width = this_scale * size
                wh_list.append((box_width, box_height))

                box_height = box_width = np.sqrt(this_scale * next_scale) * size
                wh_list.append((box_width, box_height))

            else:
                # One box for each other aspect ratio
                box_width = this_scale * size * np.sqrt(ar)
                box_height = this_scale * size / np.sqrt(ar)
                wh_list.append((box_width, box_height))

        wh_list = np.array(wh_list)
        n_boxes = len(wh_list)

        # Compute the grid of box center points. They are identical for all aspect ratios.

        # Compute the step sizes, i.e. how far apart the anchor box center points will be vertically and horizontally.
        step_height = this_steps
        step_width = this_steps

        # Compute the offsets, i.e. at what pixel values the first anchor box center point will be from the top and from the left of the image.
        offset_height = this_offsets
        offset_width = this_offsets

        ### Logic for generation of anchor boxes

        # Calculation of the centriods of anchor boxes in the feature map
        cy = np.linspace(offset_height * step_height, (offset_height + feature_map_size[0] - 1) * step_height, feature_map_size[0])
        cx = np.linspace(offset_width * step_width, (offset_width + feature_map_size[1] - 1) * step_width, feature_map_size[1])

        # converting it to 2 dimensions as the feature maps are 2D too.
        cx_grid, cy_grid = np.meshgrid(cx, cy)
        cx_grid = np.expand_dims(cx_grid, -1)
        cy_grid = np.expand_dims(cy_grid, -1)

        # Create a 4D tensor template of shape `(feature_map_height, feature_map_width, n_boxes, 4)`
        # where the last dimension will contain `(cx, cy, w, h)`
        boxes_tensor = np.zeros((feature_map_size[0], feature_map_size[1], n_boxes, 4))

        # Assigning the generated centriods and height and width to the tensor
        boxes_tensor[:, :, :, 0] = np.tile(cx_grid, (1, 1, n_boxes)) # Set cx
        boxes_tensor[:, :, :, 1] = np.tile(cy_grid, (1, 1, n_boxes)) # Set cy
        boxes_tensor[:, :, :, 2] = wh_list[:, 0] # Set w
        boxes_tensor[:, :, :, 3] = wh_list[:, 1] # Set h

        # Convert to corners
        boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion=1)

        # Normalize box coordinates
        if self.normalize_coords:

            boxes_tensor[:, :, :, [0, 2]] /= self.img_width
            boxes_tensor[:, :, :, [1, 3]] /= self.img_height

        # Convert to centroids
        boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion=2)

        if diagnostics:
            return boxes_tensor, (cy, cx), wh_list, (step_height, step_width), (offset_height, offset_width)
        else:
            return boxes_tensor

    '''
    This function initializes the variables that will store the labels in encoded form
    with a template which will further be updated with labels and anchorboxes
    '''
    def generate_encoding_template(self, batch_size, diagnostics=False):

        # Tile the anchor boxes for each predictor layer across all batch items.
        boxes_batch = []

        for boxes in self.boxes_list:
            # Prepend one dimension to `self.boxes_list` to account for the batch size and tile it along.
            # The result will be a 5D tensor of shape `(batch_size, feature_map_height, feature_map_width, n_boxes, 4)`
            boxes = np.expand_dims(boxes, axis=0)
            boxes = np.tile(boxes, (batch_size, 1, 1, 1, 1))

            # Now reshape the 5D tensor above into a 3D tensor of shape
            # `(batch, feature_map_height * feature_map_width * n_boxes, 4)`. The resulting
            # order of the tensor content will be identical to the order obtained from the reshaping operation
            # in our Keras model (we're using the Tensorflow backend, and tf.reshape() and np.reshape()
            # use the same default index order, which is C-like index ordering)
            boxes = np.reshape(boxes, (batch_size, -1, 4))
            boxes_batch.append(boxes)

        # Concatenate the anchor tensors from the individual layers to one.
        boxes_tensor = np.concatenate(boxes_batch, axis=1)

        # Create a template tensor to hold the one-hot class encodings of shape `(batch, #boxes, #classes)`
        # It will contain all zeros for now, the classes will be set in the matching process that follows
        classes_tensor = np.zeros((batch_size, boxes_tensor.shape[1], self.n_classes))

        # Create a tensor to contain the variances. This tensor has the same shape as `boxes_tensor` and simply
        # contains the same 4 variance values for every position in the last axis.
        variances_tensor = np.zeros_like(boxes_tensor)
        variances_tensor += self.variances # Long live broadcasting

        # Concatenate the classes, boxes and variances tensors to get our final template for y_encoded. We also need
        # another tensor of the shape of `boxes_tensor` as a space filler so that `y_encoding_template` has the same
        # shape as the SSD model output tensor. The content of this tensor is irrelevant, we'll just use
        # `boxes_tensor` a second time.
        y_encoding_template = np.concatenate((classes_tensor, boxes_tensor, boxes_tensor, variances_tensor), axis=2)

        if diagnostics:
            return y_encoding_template, self.centers_diag, self.wh_list_diag, self.steps_diag, self.offsets_diag
        else:
            return y_encoding_template

class DegenerateBoxError(Exception):
    '''
    An exception class to be raised if degenerate boxes are being detected.
    '''
    pass
