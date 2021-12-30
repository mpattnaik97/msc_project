from data_format import DataGenerator
from ssd import original_ssd
from ssd_v2 import another_ssd
from model_evaluator import Evaluator
from losses import SSDLoss
from keras import backend as K
import keras

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
generator = dataset.generate(batch_size=8,
                             shuffle=True,
                             returns={'processed_images',
                                      'original_images',
                                      'original_labels'},
                             keep_images_without_gt=False)

# Instantiate the evaluator object which will evalute the model and give the mAP
evaluator = Evaluator(model=model,
                      n_classes=20,
                      data_generator=dataset,
                      model_mode='training')

# call the function to evalute the dataset batch-wise and calculate the mean average precision
results = evaluator(img_height=img_height,
                    img_width=img_width,
                    batch_size=8,
                    data_generator_mode='resize',
                    round_confidences=False,
                    matching_iou_threshold=0.5,
                    border_pixels='include',
                    sorting_algorithm='quicksort',
                    average_precision_mode='sample',
                    num_recall_points=11,
                    ignore_neutral_boxes=True,
                    return_precisions=True,
                    return_recalls=True,
                    return_average_precisions=True,
                    verbose=True)

# Print the results
mean_average_precision, average_precisions, precisions, recalls = results

for i in range(1, len(average_precisions)):
    print("{:<14}{:<6}{}".format(classes[i], 'AP', round(average_precisions[i], 3)))

print("\n {:<14}{:<6}{}".format('','mAP', round(mean_average_precision, 3)))
