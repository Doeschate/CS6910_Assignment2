# CS6910_assignment2_PartA

## Problem Statement
The problem statement of this part is to build a 5 layer CNN model and train the model from scratch to classify inaturalist data. Implement different training technigues and do experiments with different hyperparameters to get maximum accuracy. Use wand.ai tool to keep track of all experiments.

## Installing Libraries

!pip install wandb  (To update experiment values to wandb).\
!pip install -U albumentations (For data augumentation).\
!pip install \"opencv-python-headless<4.3\" (For import albumentations as A).\

## Code

We created five code files for part A question.

**Assignment2_PartA_Q1.ipynb** : Code is written in notebook style. we can download the file and upload to google colab or kaggle and can run
all the cells.\
The main() function contains these following variables which can be manually changed for different combination of parameters of the model which makes the code flexible such that the number of filters, size of filters and activation function in each layer can be changed along with the number of neurons in the dense layer.
The parameters are:\
##Attributes for 1st Convolution Layer\
conv_attributes[0]["out_channels"]=6\
conv_attributes[0]["kernel_size"]=11

##Attributes for 2nd Convolution Layer\
conv_attributes[1]["out_channels"]=12\
conv_attributes[1]["kernel_size"]=9

##Attributes for 3rd Convolution Layer\
conv_attributes[2]["out_channels"]=16\
conv_attributes[2]["kernel_size"]=7

##Attributes for 4th Convolution Layer\
conv_attributes[3]["out_channels"]=32\
conv_attributes[3]["kernel_size"]=5

##Attributes for 5th Convolution Layer\
conv_attributes[4]["out_channels"]=32\
conv_attributes[4]["kernel_size"]=3

#dense layer size\
dense_layer_size = 32\
activation_name = 'relu'

**Assignment2_PartA_Q1.py**
This file contains Part A Question 1 code to run it in command line by typing the command\

 python Assignment2_PartA_Q1.py conv_out_channel_0, conv_kernel_size_0, conv_out_channel_1, conv_kernel_size_1,conv_out_channel_2, conv_kernel_size_2,conv_out_channel_3, conv_kernel_size_3,conv_out_channel_4, conv_kernel_size_4,dense_layer_size,activation_name
 
 Here conv_out_channel_i are the number of filters used in that layer and conv_kernel_size_i are the sizes of the filters in that layer. dense_layer_size is the argument for number of neurons in dense layer and activation_name is to choose the required activation function.
 

**Assignment2_PartA_Q2.ipynb** :

**Assignment2_PartA_Q4.ipynb** :

**Assignment2_PartA_Q5.ipynb** :

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
The report for this assignment : [link]().

## Authors

 - [Kondapalli Jayavardhan](https://github.com/jayavardhankondapalli) 
 - [Prithaj Banerjee](https://github.com/Doeschate)
