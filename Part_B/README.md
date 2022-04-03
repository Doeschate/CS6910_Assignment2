# CS6910_assignment1

## Problem Statement

The problem statement of this part is to implement code to classify inaturalist data using pretrained models. Implement different training technigues and do experiments with different hyperparameters to get maximum accuracy. Use wand.ai tool to keep track of all experiments.

## Installing Libraries

!pip install wandb  (To update experiment values to wandb).\
!pip install -U albumentations (For data augumentation).\
!pip install \"opencv-python-headless<4.3\" (For import albumentations as A).\
!pip install timm (For InceptionResNetV2 pretrained model).

## Code

We created two code files for part B question.

**DL_Assignment2_partB.ipynb** : Code is written in notebook style. we can download the file and upload to google colab or kaggle and can run
all the cells. wandb sweep is implemeted in this notebook. SO, when we run this notebook, all the necessary libraries are installed and
dataset is downloaded and sweep is activated to run all combinations of hyperparameters. Plots are generated in the wandb.
Dataset is automatically downloaded to colab or kaggle by using wget command (code written in first cell). 

**DL_Assignment2_partB_code_using_commandLineArguments.py** : In this file code is written in a way to execute in a local machine and can send the
parameters as a command line arguments. The order of command line arguments are :\
model (Name of the pretrained model)\
epochs (Number of epochs to train the model)\
learning_rate\
batch_size\
weight_decay\
unfreezed_from_last  (Number of layers from last to update the weights during training)\
dataset_augmentation (Boolean value , If True data augumentation will done otherwise not)

Paste the local path of dataset in the actual_data_path variable in the code.
If dataset is locates at "C:/nature_12K/inaturalist_12K/" then give  actual_data_path = "C:/nature_12K/inaturalist_12K/". Code will take care to load
train, validation, test data.

## Report
The report for this assignment : [link](https://wandb.ai/cs21s045_cs21s011/uncategorized/reports/Assignment-2--VmlldzoxNzY2NTQz).

## Authors

 - [Kondapalli Jayavardhan](https://github.com/jayavardhankondapalli) 
 - [Prithaj Banerjee](https://github.com/Doeschate)
