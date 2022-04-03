# All import statements

import sys,os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets,models
import torchvision.transforms as T
from PIL import Image as image1
import glob
from torch.optim import lr_scheduler
import time
import copy
from itertools import chain
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
import random
import gc
import timm
import copy
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import cv2

%matplotlib inline
%config InlineBackend.figure_format = 'retina'
import matplotlib.pyplot as plt
from itertools import chain

actual_data_path = "/content/drive/MyDrive/nature_12K/inaturalist_12K/"

#Create dictionary for class indexes
train_data_path = os.path.join(actual_data_path, "train")
classes = [] #to store class values
for data_path in glob.glob(train_data_path + "/*"):
    classes.append(data_path.split('/')[-1])
idx_to_class = {i:j for i, j in enumerate(classes)} #index to class map
class_to_idx = {value:key for key,value in idx_to_class.items()} #class to index map

# get all the paths from train_data_path and returns image paths for train and validation set
def CreateTrainDataset(actual_data_path):
    train_data_path = os.path.join(actual_data_path, "train")
    train_image_paths = [] #to store image paths in list
    classes = [] #to store class values
    for data_path in glob.glob(train_data_path + "/*"):
        train_image_paths.append(glob.glob(data_path + '/*')) #stores all the training image paths in this list
    train_image_paths = list(chain.from_iterable(train_image_paths))
    random.shuffle(train_image_paths)

    # split train valid from train paths (90,10)
    train_image_paths, valid_image_paths = train_image_paths[:int(0.9*len(train_image_paths))], train_image_paths[int(0.9*len(train_image_paths)):] 
    return train_image_paths, valid_image_paths

# create the test_image_paths
def CreateTestDataset(actual_data_path):
    test_data_path = os.path.join(actual_data_path, "val")
    test_image_paths = []
    for data_path in glob.glob(test_data_path + '/*'):
        test_image_paths.append(glob.glob(data_path + '/*')) #stores all the test images path in this list
    test_image_paths = list(chain.from_iterable(test_image_paths))
    return test_image_paths

#Function returns images and corresponding lebels after performing transforms
class iNaturalist_12KDataset(Dataset):
    def __init__(self, image_paths, transform=False):
        self.image_paths = image_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        label = image_filepath.split('/')[-2]
        label = class_to_idx[label]
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image, label
    
# all input images are resized reshized_shape   
resized_shape = 256

# Data augumentation transform functions
augmented_transforms = A.Compose([A.SmallestMaxSize(max_size=350),
              A.Resize(resized_shape,resized_shape),
              A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=360, p=0.5),
              A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
              A.RandomBrightnessContrast(p=0.5),
              A.MultiplicativeNoise(multiplier=[0.5,2], per_channel=True, p=0.2),
              A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
              A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
              A.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
              ToTensorV2()])

# train_model will train the given pretrained model with TrainDataset with num_epochs and return the best model with weights 
def train_model(model, criterion, optimizer, num_epochs, batch_size, TrainDataset, ValDataset):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, int(np.ceil(len(DataLoader(TrainDataset, batch_size,shuffle=True).dataset)/batch_size)))
    
    for epoch in range(num_epochs):  # Training will done for num_epochs times
        epochStartTime = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        datasize = 0

        model.train()   # Model in training mode
        running_loss = 0.0
        running_corrects = 0

        for data in DataLoader(TrainDataset, batch_size,shuffle=True): # Model is trained with batch size images at a time
            inputs,labels = data      
            optimizer.zero_grad()       # making accumulated gradients to zero for every batch
            
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()
                scheduler.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                datasize += inputs.shape[0]

        training_epoch_loss = running_loss / datasize
        training_epoch_acc = running_corrects.double() / datasize

        print('{} Loss: {:.4f} Acc: {:.4f}'.format('Train', training_epoch_loss, training_epoch_acc))

        datasize = 0
        model.eval()  # setting model in evalution mode
        running_loss = 0.0
        running_corrects = 0

        for data in DataLoader(ValDataset,batch_size):
            inputs,labels = data

            optimizer.zero_grad()

            with torch.set_grad_enabled(False):  # gradients will not calculated (As we are validating the model)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            datasize += inputs.shape[0]

        validation_epoch_loss = running_loss / datasize
        validation_epoch_acc = running_corrects.double() / datasize

        print('{} Loss: {:.4f} Acc: {:.4f}'.format('Val', validation_epoch_loss, validation_epoch_acc))

        if validation_epoch_acc > best_acc: # saving the model weights if this epoch has best validation accuracy
            best_acc = validation_epoch_acc 
            best_model_wts = copy.deepcopy(model.state_dict())
        print()

        
        epochEndTime = time.time() - epochStartTime
        print('Epoch complete in {:.0f}m {:.0f}s'.format(
                epochEndTime // 60, epochEndTime % 60))
            

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
    model.load_state_dict(best_model_wts)  # returning model with best weights
    return model

# test model will take Dataloader object, model, batch_size as input and returns accuracy as output
def test_model(model, TestDataset,batch_size):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in DataLoader(TestDataset,batch_size):
            images, labels = data

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the test images: {100 * correct // total} %')
  
def main(model,epochs,learning_rate,batch_size,weight_decay,unfreezed_from_last,dataset_augmentation):
    resized_shape = 256
   
    # preparing Training Data
    train_transforms = A.Compose([A.Resize(resized_shape,resized_shape),A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),ToTensorV2()])
    train_image_paths, valid_image_paths=CreateTrainDataset(actual_data_path)
    TrainDataset = iNaturalist_12KDataset(train_image_paths,train_transforms)
    
    # Increasing Training Dataset by doing data augumentation 
    if dataset_augmentation:
        transformed_dataset = iNaturalist_12KDataset(train_image_paths,augmented_transforms)   #Transformed Dataset created with augmented_transforms
        TrainDataset = torch.utils.data.ConcatDataset([transformed_dataset,TrainDataset])
    
    # preparing Validation and Test dataset
    ValDataset = iNaturalist_12KDataset(valid_image_paths,train_transforms)
    test_transforms = A.Compose([A.Resize(resized_shape,resized_shape),A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),ToTensorV2()])
    test_image_paths=CreateTestDataset(actual_data_path)
    TestDataset = iNaturalist_12KDataset(test_image_paths,test_transforms)
    
    if model == 'InceptionV3':
        model = models.inception_v3(pretrained=True,aux_logits=False)
    elif model == 'resnet50':
        model = models.resnet50(pretrained=True)
    elif model == 'resnet18':
        model = models.resnet18(pretrained=True)
    elif model == 'googlenet':
        model = models.googlenet(pretrained=True,aux_logits=False)
    elif model == 'InceptionResNetV2':
        model = timm.create_model('inception_resnet_v2', pretrained=True)
    
    # Freezing the initial layers of model according to unfreezed_from_last parameter
    for param in list(model.parameters())[: -(unfreezed_from_last+2)]:
        param.requires_grad = False

    # Changing final layer neurons count to 10 (As our inout dataset has 10 labels)
    if model == 'InceptionResNetV2':
        num_ftrs = model.classif.in_features
        model.classif = nn.Linear(num_ftrs, 10)
    else:
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 10)
    
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.Adam(model.parameters(), lr=learning_rate,weight_decay = weight_decay)

    # Training the model
    model = train_model(model, criterion, optimizer_ft, epochs, batch_size, TrainDataset, ValDataset)
    #Testing the model
    test_model(model, TestDataset, batch_size)


if  __name__ =="__main__":
  
  n = len(sys.argv) # number of command line arguments passed
  
  model = sys.argv[1]
  epochs = int(sys.argv[2])
  learning_rate = float(sys.argv[3])
  batch_size = int(sys.argv[4])
  weight_decay = float(sys.argv[5])
  unfreezed_from_last = int(sys.argv[6])
  dataset_augmentation = sys.argv[7]

  main(model, epochs, learning_rate, batch_size, weight_decay, unfreezed_from_last, dataset_augmentation)
  
