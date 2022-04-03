# CS6910_assignment2_PartA

## Problem Statement
The problem statement of this part is to build a 5 layer CNN model and train the model from scratch to classify inaturalist data. Implement different training technigues and do experiments with different hyperparameters to get maximum accuracy. Use wand.ai tool to keep track of all experiments.

## Installing Libraries

!pip install wandb  (To update experiment values to wandb).\
!pip install -U albumentations (For data augumentation).\
!pip install \"opencv-python-headless<4.3\" (For import albumentations as A).\

## Code

We created eight code files for part A question.

**Assignment2_PartA_Q1.ipynb** :\
Code is written in notebook style. we can download the file and upload to google colab or kaggle and can run
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

**Assignment2_PartA_Q1.py**:\
This file contains Part A Question 1 code to run it in command line by typing the command\

 python Assignment2_PartA_Q1.py conv_out_channel_0, conv_kernel_size_0, conv_out_channel_1, conv_kernel_size_1,conv_out_channel_2, conv_kernel_size_2,conv_out_channel_3, conv_kernel_size_3,conv_out_channel_4, conv_kernel_size_4,dense_layer_size,activation_name
 
 Here conv_out_channel_i are the number of filters used in that layer and conv_kernel_size_i are the sizes of the filters in that layer. dense_layer_size is the argument for number of neurons in dense layer and activation_name is to choose the required activation function.
 

**Assignment2_PartA_Q2.ipynb** :\
This notebook can be uploaded in kaggle or google colab and the cells can be run one after another as in the order to train using sweep parameters and generate the plots and find the test accuracy also.\
actual_data_path = "./inaturalist_12K" is path set for kaggle.\
To run in colab replace the path accordingly\
Run upto the cell given below to run and plot wandb graphs\
#Run this cell to start sweep\
wandb.agent(sweep_id, train_wandb , project="Assignment2_PartA",count=5)\
wandb.finish()\
Run the main() function instead to train and test with given paramters by flexible changing the code

**Assignment2_PartA_Q4.ipynb** :\
This notebook can be uploaded in kaggle or google colab and the cells can be run one after another as in the order to train using sweep parameters for the best model whose parameters are stored in sweep_config and generate the plots using the functions PlotGridOfImages(model,batch_size): and visualize_filters(model,batch_size): The required images will be plot after this.



**Assignment2_PartA_Q5.ipynb** :\
This notebook can be uploaded in kaggle or colab and the cells can be run one after another as in the order to get the guided back propagation on any 10 neurons in the CONV5 layer and plot the images which excite this neuron.

**Assignment2_PartA.ipynb** contains all the Assignment2_PartA codes in one notebook for reference.It is modified version of Pytorch_Assignment2.ipynb


## Report
The report for this assignment : [link](https://wandb.ai/cs21s045_cs21s011/uncategorized/reports/Assignment-2--VmlldzoxNzY2NTQz).

## Authors

 - [Kondapalli Jayavardhan](https://github.com/jayavardhankondapalli) 
 - [Prithaj Banerjee](https://github.com/Doeschate)
