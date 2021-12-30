from data_format import DataGenerator
from ssd import original_ssd
from ssd_v2 import another_ssd
from losses import SSDLoss
from keras import backend as K
import keras
from output_decoder import decode_detections
import numpy as np
import matplotlib.pyplot as plt

K.clear_session() # Clear previous models from memory.

# change the image height and width to 300 for original_ssd
img_height = 345 # Height of the model input images
img_width = 345 # Width of the model input images

# Instantiate the model
# model = original_ssd() # uncomment if using original_ssd

model = another_ssd() # comment if using original_ssd

# load up the weights of the model for evaluation
weights_path = '../saved_models/another_ssd_weights_80.hdf5'
model.load_weights(weights_path, by_name=True)

# Compile the model so that Keras won't complain the next time you load it.
adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

# Instantiate the DataGenerator to generate data in batches
dataset = DataGenerator(out_width=img_width, out_height=img_height)

# set the path to required files and directories
VOC_2007_images_dir         = '../VOCdevkit/VOC2007/JPEGImages/'
VOC_2007_annotations_dir    = '../VOCdevkit/VOC2007/Annotations/'
VOC_2007_test_image_set_filename = '../VOCdevkit/VOC2007/ImageSets/Main/test.txt'


# List of class names where the indices are their respective class IDs
classes = ['background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat',
           'chair', 'cow', 'diningtable', 'dog',
           'horse', 'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']

# Load up the class variables with image information and labels
dataset.parse_xml(images_dirs=[VOC_2007_images_dir],
                  image_set_filenames=[VOC_2007_test_image_set_filename],
                  annotations_dirs=[VOC_2007_annotations_dir],
                  classes=classes,
                  include_classes='all')

# Get the generator object from the DataGenerator object
generator = dataset.generate(batch_size=1,
                             shuffle=True,
                             returns={'processed_images',
                                      'original_images',
                                      'original_labels',
                                      'image_sizes',
                                      'image_ids'},
                             keep_images_without_gt=False)

# Loop to generate predictions and plot them on images. Loops runs based on input by user
while True:

    # Generate data with batch size 1
    images, image_ids, original_images, labels, image_size = next(generator)

    # generate predictions
    preds = model.predict_on_batch(images)

    # convert the predictions in the format similar to ground truth
    results = decode_detections(preds,
                                confidence_thresh=0.01,
                                iou_threshold=0.45,
                                top_k=200,
                                input_coords='centroids',
                                normalize_coords=True,
                                img_height=img_height,
                                img_width=img_width,
                                border_pixels='half')


    # Select the best predictions from the 200 predictions produced by the above function
    confidence_threshold = 0.5

    results = np.array(results)
    results_thresh = [results[k][results[k,:,1] > confidence_threshold] for k in range(results.shape[0])]

    # apply inverse transforms
    resized_res = []
    original_size = np.array(image_size[0], dtype=np.float)

    for r in results_thresh:
        resized_r = []
        for box in r:
            box[[2,4]] = np.round(box[[2,4]] * (original_size[1]/img_width))
            box[[3,5]] = np.round(box[[3,5]] * (original_size[0]/img_height))
            resized_r.append(r)
        resized_res.append(r)

    # plot the class names, bounding boxes on image
    # ground truth are green and predictions are red
    plt.figure(figsize=(20,12))
    plt.imshow(original_images[0])

    current_axis = plt.gca()

    for box in labels[0]:
        xmin = box[1]
        ymin = box[2]
        xmax = box[3]
        ymax = box[4]
        label = '{}'.format(classes[int(box[0])])
        current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color='green', fill=False, linewidth=2))
        current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':'green', 'alpha':1.0})

    for box in resized_res[0]:
        xmin = box[2]
        ymin = box[3]
        xmax = box[4]
        ymax = box[5]
        label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
        current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color='red', fill=False, linewidth=2))
        current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':'red', 'alpha':1.0})

    plt.show()

    if input('any key to continue, q to quit:') == 'q':
        break
