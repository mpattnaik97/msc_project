import cv2
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
img_height = 300 # Height of the model input images
img_width = 300 # Width of the model input images

# img_height = 345 # Height of the model input images
# img_width = 345 # Width of the model input images

# Instantiate the model
model = original_ssd() # uncomment if using original_ssd

# model = another_ssd() # comment if using original_ssd

# load up the weights of the model for evaluation
weights_path = '../saved_models/original_ssd_weights_80.hdf5'

# weights_path = '../saved_models/another_ssd_weights_80.hdf5'

model.load_weights(weights_path, by_name=True)

# Compile the model so that Keras won't complain the next time you load it.
adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
model.compile(optimizer=adam, loss=ssd_loss.compute_loss)


# List of class names where the indices are their respective class IDs
classes = ['background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat',
           'chair', 'cow', 'diningtable', 'dog',
           'horse', 'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']

# Loop to generate predictions and plot them on images. Loops runs based on input by user
import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, image = cap.read()

    resized_image = cv2.resize(image, (img_height,img_width), interpolation = cv2.INTER_AREA)

    image_tensor = np.expand_dims(resized_image, axis=0)
    # generate predictions
    preds = model.predict(image_tensor)

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
    original_size = image.shape[:2]

    for r in results_thresh:
        resized_r = []
        for box in r:
            box[[2,4]] = np.round(box[[2,4]] * (original_size[1]/img_width))
            box[[3,5]] = np.round(box[[3,5]] * (original_size[0]/img_height))

            xmin = box[2]
            ymin = box[3]
            xmax = box[4]
            ymax = box[5]

            text = classes[int(box[0])]
            text = text + ':' + str(round(box[1], 2))
            image = cv2.rectangle(img=image, pt1=(int(xmin), int(ymin)), pt2=(int(xmax), int(ymax)), color=(0, 255, 0), thickness=3)

            (w, h) , baseline = cv2.getTextSize(text, fontFace= cv2.FONT_HERSHEY_COMPLEX, fontScale=0.7, thickness=2)
            image = cv2.rectangle(img=image, pt1=(int(xmin - 2), int(ymin - 10 - h)), pt2=(int(xmin + w), int(ymin)), color=(0, 255, 0), thickness=-1)

            image = cv2.putText(img=image, text=text, org=(int(xmin), int(ymin - 10)), fontFace= cv2.FONT_HERSHEY_COMPLEX, fontScale=0.7, color=(255, 255, 255), thickness=2)

    # Display the resulting frame
    cv2.imshow('frame', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
