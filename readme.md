[Note: The link for downloading datasets and model weights can be found in the file sopporting_materials.pdf provided in this supporting materials folder] 

The supporting material contains 2 directories namely:

1. source_code: which contains the implemented code.
3. results: contains the logs of training, i.e. record of training and validation loss at different epochs.

[Note: Two more directories will be added named "VOCdevkit" and "saved_model" after you download and extract your dataset and model weights here]

Along with that it also has a file names supporting_materials.pdf which contains the link to download datasets and model weights.
[Note: you will find the instructions for downloading and setting up the dataset and model weights in the pdf along with the link to download them]

[Note: All the code is well commented for use.]

The main contents of source code directory are as follows:

1. main_training.py - This contains the code to train the original ssd model.
2. another_training.py - This trains the modified ssd model which I call another ssd model.
3. inference.py - It will help you visualize the results by plotting the class names and bounding boxes on the image.
4. eval.py - It calculates the mAP of the model over test dataset and gives you the results.
5. plot_results - saves a graph of training loss and validation loss against number of epochs to loss.png

python libraries required to run the scripts are -

1. numpy
2. pandas
3. tensorflow-gpu 1.13.1
4. keras 2.2.4
5. matplotlib
6. scipy
7. pillow
8. math
9. bs4
10. pickle
11. copy
12. opencv
13. os
14. sys
15. pdb
16. scikit-learn
17. tqdm
18. warnings

Steps to use main_training.py -
1. On lines 29-40 are defined the path to datasets and directories needed for training. This can be changed if needed.
2. On line 69, I have set the batch size to 10 to avoid out of memory errors. If you are using advanced hardware, I'll recommend using batch_size = 32.
3. Lines 97 and 98 let you load model weights to the the model. It will be commented, to load up the model with weights, uncomment the lines and provide the path to the weights file in the path = ' ' on line 97.
4. Line 126 stores the result logs in csv format. You can modify the path you want the csv file to be as per your needs
5. line  129 saves models in the given path at intervals specified in the 'period' argument. It is set to 20 epochs, but you can change it if you want. 
6. After doing all the above steps, run the python file and the model will start training.

Steps to use another_training.py -
1. On lines 29-40 are defined the path to datasets and directories needed for training. This can be changed if needed.
2. On line 69, I have set the batch size to 10 to avoid out of memory errors. If you are using advanced hardware, I'll recommend using batch_size = 32.
3. Lines 98 and 99 let you load model weights to the the model. It will be commented, to load up the model with weights, uncomment the lines and provide the path to the weights file in the path = ' ' on line 97.
4. Line 127 stores the result logs in csv format. You can modify the path you want the csv file to be as per your needs
5. line  130 saves models in the given path at intervals specified in the 'period' argument. It is set to 20 epochs, but you can change it if you want. 
6. After doing all the above steps, run the python file and the model will start training.

Steps to use eval.py -
1. If you are performing evaluation on the original ssd model, change the img_height and img_width values to 300 on lines 12 and 13. Also uncomment line 17 and comment out line 19.
2. If evaluating another ssd model, no need to do anything.
3. Provide the path to the weight file on line 22 which you want to evalute. [Note: loading the weights of original ssd on another ssd and vice versa would result in an error]
4. Run the python file and it will provide you with mAP results for model with the loaded weights.

Steps to use inference.py -
1. If you are performing evaluation on the original ssd model, change the img_height and img_width values to 300 on lines 14 and 15. Also uncomment line 18 and comment out line 20.
2. If evaluating another ssd model, no need to do anything.
3. Provide the path to the weight file on line 23 which you want to evalute. [Note: loading the weights of original ssd on another ssd and vice versa would result in an error]
4. Run the python file and it will provide you with visual results for model with the loaded weights.

Steps to use plot_results.py -
1. On line 5, give the path of the file containing the log of results you want the plot for.
2. Run the script.

******
[Note: The python scripts that are referenced below are those scripts which I did not create nor did I modify them much. These are codes that I reused under the Apache license for the project and all the credit for making it goes to the original developers] 
******

Supplementary contents of the source code directory:
1. anchorboxes.py - defines a keras layer to generate anchor boxes for the predictor layers.
2. box_validation_utils.py (Ferrari, 2018) - defines class and functions responsible for checking if a ground truth bounding box is valid. If not, that data item is dropped and not used in training.
3. data_format.py - responsible for reading, parsing and encoding of the input data to make it fit for the SSD neural network training, validation or testing.
4. encode_input.py - responsible for encoding the ground truth labels in a way the loss function can use to train the model.
5. losses.py - defines functions to calculate the loss function for SSD.
6. model_evaluator.py (Ferrari, 2018) - defines class and function for calculating the mean average precision of a model on a dataset.
7.normalize.py (Ferrari, 2018) - defines a keras layer for L2 normalization.
8. output_decoder.py (Ferrari, 2018) - this script is responsible for converting the predicted labels to a format similar to the ground truth values. Also, it is responsible for performing non maximum supression and 
		confidence thresholding for selecting 200 top predictions out of 8732 predictions per image.
9. ssd.py - defines a function that creates the original ssd model
10. ssd_v2.py - defines a function that generates another ssd model (having my modifications over original model).

References -
1. Ferrari, P. 2018, ssd_keras, https://github.com/pierluigiferrari/ssd_keras
2. Vinodbabu, S. 2019 a-PyTorch-Tutorial-to-Object-Detection, https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection 