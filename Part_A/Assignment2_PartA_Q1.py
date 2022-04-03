import sys,os
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
#Activation Function
#To add another activation just add another else statement for that activation and return the corresponding pytorch reference for that activation
def ActivationFunction(activation_name):
    if(activation_name == 'relu'):
        return F.relu  
    elif(activation_name == 'elu'):
        return F.elu
    elif(activation_name == 'sigmoid'):
        return F.sigmoid
    elif(activation_name == 'gelu'):
        return F.gelu
    else:
        return None

# Build a small CNN model consisting of  5 convolution layers.
# Each convolution layer would be followed by a ReLU activation and a max pooling layer.
# After 5 such conv-relu-maxpool blocks of  layers you should have one dense layer followed by the output layer containing 10 neurons (1 for each of the 10 classes).

class CnnModel(nn.Module):
    def __init__(self, conv_attributes, pool_attributes,in_feature,dense_layer_size,activation_name):
        super(CnnModel, self).__init__()

        #First Convolution and Pooling Layer
        self.conv1= nn.Conv2d(conv_attributes[0]["in_channels"], conv_attributes[0]["out_channels"], conv_attributes[0]["kernel_size"])
        self.act1 = ActivationFunction(activation_name)
        self.pool1= nn.MaxPool2d(pool_attributes[0]["kernel_size"], pool_attributes[0]["stride"])

        #Second Convolution and Pooling Layer
        self.conv2= nn.Conv2d(conv_attributes[1]["in_channels"], conv_attributes[1]["out_channels"], conv_attributes[1]["kernel_size"])
        self.act2 = ActivationFunction(activation_name)  
        self.pool2= nn.MaxPool2d(pool_attributes[1]["kernel_size"], pool_attributes[1]["stride"])

        #Third Convolution and Pooling Layer
        self.conv3= nn.Conv2d(conv_attributes[2]["in_channels"], conv_attributes[2]["out_channels"], conv_attributes[2]["kernel_size"])
        self.act3 = ActivationFunction(activation_name) 
        self.pool3= nn.MaxPool2d(pool_attributes[2]["kernel_size"], pool_attributes[2]["stride"])

        #Fourth Convolution and Pooling Layer
        self.conv4= nn.Conv2d(conv_attributes[3]["in_channels"], conv_attributes[3]["out_channels"], conv_attributes[3]["kernel_size"])
        self.act4 = ActivationFunction(activation_name) 
        self.pool4= nn.MaxPool2d(pool_attributes[3]["kernel_size"], pool_attributes[3]["stride"])

        #Fifth Convolution and Pooling Layer
        self.conv5= nn.Conv2d(conv_attributes[4]["in_channels"], conv_attributes[4]["out_channels"], conv_attributes[4]["kernel_size"])
        self.act5 = ActivationFunction(activation_name) 
        self.pool5= nn.MaxPool2d(pool_attributes[4]["kernel_size"], pool_attributes[4]["stride"])

        #First Dense Layer
        self.fc1 = nn.Linear(in_feature, dense_layer_size)
        self.fc1_act = ActivationFunction(activation_name)
        self.fc2 = nn.Linear(dense_layer_size, 10)

    def forward(self,x):
        
        x = self.pool1(self.act1(self.conv1(x))) #First block of layer containing one conv layer with  activation function followed by one pooling layer
        x = self.pool2(self.act2(self.conv2(x))) #Second block of layer containing one conv layer with  activation function followed by one pooling layer
        x = self.pool3(self.act3(self.conv3(x))) #Third block of layer containing one conv layer with  activation function followed by one pooling layer
        x = self.pool4(self.act4(self.conv4(x))) #Fourth block of layer containing one conv layer with  activation function followed by one pooling layer
        x = self.pool5(self.act5(self.conv5(x))) #Fifth block of layer containing one conv layer with  activation function followed by one pooling layer

        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc1(x)
        x = self.fc1_act(x)
        x = self.fc2(x)
        x = F.softmax(x,dim=1)                     
        return x

##Calculates the input feature for the dense linear layer
def LinearInFeatureCalculate(initial_dim,conv_attributes,pool_attributes):
    for i in range(5):
        D = (initial_dim + 2*conv_attributes[i]["padding"] - conv_attributes[i]["dilation"]*(conv_attributes[i]["kernel_size"]-1) - 1)//(conv_attributes[i]["stride"]) + 1
        D = (D - pool_attributes[i]["kernel_size"])//(pool_attributes[i]["stride"]) + 1
        initial_dim = D
    return D

#Main function 
def main(conv_out_channel_0, conv_kernel_size_0, conv_out_channel_1, conv_kernel_size_1,conv_out_channel_2, conv_kernel_size_2,conv_out_channel_3, conv_kernel_size_3,conv_out_channel_4, conv_kernel_size_4,dense_layer_size,activation_name):
    resized_shape = 256

    conv_attributes = [{"in_channels":0,"out_channels":0,"kernel_size":0, "stride":1, "padding":0, "dilation":1},
                     {"in_channels":0,"out_channels":0,"kernel_size":0, "stride":1, "padding":0, "dilation":1},
                     {"in_channels":0,"out_channels":0,"kernel_size":0, "stride":1, "padding":0, "dilation":1},
                     {"in_channels":0,"out_channels":0,"kernel_size":0, "stride":1, "padding":0, "dilation":1},
                     {"in_channels":0,"out_channels":0,"kernel_size":0, "stride":1, "padding":0, "dilation":1}]


    pool_attributes = [{"kernel_size":1, "stride": 1},
                     {"kernel_size":1, "stride": 1},
                     {"kernel_size":1, "stride": 1},
                     {"kernel_size":1, "stride": 1},
                     {"kernel_size":1, "stride": 1}]

    ##Attributes for 1st Convolution Layer
    conv_attributes[0]["in_channels"]=3
    conv_attributes[0]["out_channels"]=conv_out_channel_0
    conv_attributes[0]["kernel_size"]=conv_kernel_size_0

    ##Attributes for 2nd Convolution Layer
    conv_attributes[1]["in_channels"]=conv_attributes[0]["out_channels"]
    conv_attributes[1]["out_channels"]=conv_out_channel_1
    conv_attributes[1]["kernel_size"]=conv_kernel_size_1

    ##Attributes for 3rd Convolution Layer
    conv_attributes[2]["in_channels"]=conv_attributes[1]["out_channels"]
    conv_attributes[2]["out_channels"]=conv_out_channel_2
    conv_attributes[2]["kernel_size"]=conv_kernel_size_2

    ##Attributes for 4th Convolution Layer
    conv_attributes[3]["in_channels"]=conv_attributes[2]["out_channels"]
    conv_attributes[3]["out_channels"]=conv_out_channel_3
    conv_attributes[3]["kernel_size"]=conv_kernel_size_3

    ##Attributes for 5th Convolution Layer
    conv_attributes[4]["in_channels"]=conv_attributes[3]["out_channels"]
    conv_attributes[4]["out_channels"]=conv_out_channel_4
    conv_attributes[4]["kernel_size"]=conv_kernel_size_4

    ##Attributes for 1st Pooling Layer
    pool_attributes[0]["kernel_size"]=2
    pool_attributes[0]["stride"]=2

    ##Attributes for 2nd Pooling Layer
    pool_attributes[1]["kernel_size"]=2
    pool_attributes[1]["stride"]=2

    ##Attributes for 3rd Pooling Layer
    pool_attributes[2]["kernel_size"]=2
    pool_attributes[2]["stride"]=2

    ##Attributes for 4th Pooling Layer
    pool_attributes[3]["kernel_size"]=2
    pool_attributes[3]["stride"]=2

    ##Attributes for 5th Pooling Layer
    pool_attributes[4]["kernel_size"]=2
    pool_attributes[4]["stride"]=2

    #dense layer size
    dense_layer_size = dense_layer_size
    activation_name = activation_name

    ##Calculating the input dimension for the Dense Linear layer
    final_dim=LinearInFeatureCalculate(resized_shape,conv_attributes,pool_attributes) #height,width of the dense layer
    in_feature = (final_dim ** 2) * conv_attributes[4]["out_channels"] #number of input nodes in the dense layer

    model = CnnModel(conv_attributes, pool_attributes,in_feature,dense_layer_size,activation_name)
    print("The model built is:")
    print(model)
    #Deleting the model after use
    del model
    gc.collect()
    torch.cuda.empty_cache()
if  __name__ =="__main__":
    
    n = len(sys.argv) # number of command line arguments passed
    
    ##Attributes for 1st Convolution Layer
    conv_out_channel_0=sys.argv[1]
    conv_kernel_size_0=sys.argv[2]

    ##Attributes for 2nd Convolution Layer
    conv_out_channel_1=sys.argv[3]
    conv_kernel_size_1=sys.argv[4]

    ##Attributes for 3rd Convolution Layer
    conv_kernel_size_2=sys.argv[6]
    conv_kernel_size_2=sys.argv[7]

    ##Attributes for 4th Convolution Layer
    conv_kernel_size_3=sys.argv[8]
    conv_kernel_size_3=sys.argv[9]

    ##Attributes for 5th Convolution Layer
    conv_kernel_size_4=sys.argv[10]
    conv_kernel_size_4=sys.argv[11]

    #dense layer size
    dense_layer_size = sys.argv[12]
    activation_name = sys.argv[13]

    main(conv_out_channel_0, conv_kernel_size_0, conv_out_channel_1, conv_kernel_size_1,conv_out_channel_2, conv_kernel_size_2,conv_out_channel_3, conv_kernel_size_3,conv_out_channel_4, conv_kernel_size_4,dense_layer_size,activation_name)