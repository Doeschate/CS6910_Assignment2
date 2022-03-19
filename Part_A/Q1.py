import sys,os
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets
import torchvision.transforms as T

from PIL import Image as image1


# Reading  image names from the all folders
root = "/content/drive/MyDrive/nature_12K/inaturalist_12K/train"
classNames = []
trainDataFiles = []
for path, subdirs, files in os.walk(root):
  classNames.append(path.split('/')[-1])
  trainDataFiles.append(files)
trainDataFiles = trainDataFiles[1:]
classNames = classNames[1:]
classes = {classNames[i] : i for i in range(0,len(classNames))}
#Reading image data and converting to numpy arrays

trainingData = []
labels = []
count = 0
for i in range(0,2):
  print(i)
  for name in trainDataFiles[i]:
    img = image1.open(root+'/'+classNames[i]+'/'+ name)
    trainingData.append(np.array(img))
    labels.append(classes[classNames[i]])
trainingData = np.array(trainingData)

class iNatureDataset(Dataset):
    def __init__(self, trainX,trainY):
        self.trainX = trainX
        self.trainY = trainY

    def __len__(self):
        return len(self.trainY)

    def __getitem__(self, index):
        image = self.trainX[index]
        label = self.trainY[index]
        X = self.transform(image)
        return [X,label]
        
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((800,800)),
        T.ToTensor()])
    
    #conv2d(input_channels, output_channels, kernel_size)
class CNN(nn.Module):
  def __init__(self):
    super().__init__()
    
    self.conv1=nn.Conv2d(3, 2, 3)
    self.pool=nn.MaxPool2d(2,2)
    
    self.conv2=nn.Conv2d(2, 2, 3)
    self.conv3=nn.Conv2d(2, 2, 3)
    self.conv4=nn.Conv2d(2, 2, 3)
    self.conv5=nn.Conv2d(2, 2, 3)

    self.fc1 = nn.Linear(1058,15)
    self.fc2 = nn.Linear(15,2)
    

  def forward(self,x):
    '''
    print('x.shape',x.shape)
    x = self.pool(F.relu(self.conv1(x)))
    print('c1',x.shape)
    '''
    #print("x.shape",x.shape)
    c1 = self.conv1(x)
    #print("c1.shape",c1.shape)
    r1 = F.relu(c1)
    #print("r1.shape",r1.shape)
    p1 = self.pool(r1) 
    #print("p1.shape",p1.shape)
    
    #x = self.pool(F.relu(self.conv2(x)))
    
    c1 = self.conv2(p1)
    #print("c2.shape",c1.shape)
    r1 = F.relu(c1)
    #print("r2.shape",r1.shape)
    p1 = self.pool(r1) 
    #print("p2.shape",p1.shape)

    #x = self.pool(F.relu(self.conv3(p1)))
    
    c1 = self.conv3(p1)
    #print("c3.shape",c1.shape)
    r1 = F.relu(c1)
    #print("r3.shape",r1.shape)
    p1 = self.pool(r1) 
    #print("p3.shape",p1.shape)
    
    #x = self.pool(F.relu(self.conv4(x)))
    
    c1 = self.conv4(p1)
    #print("c4.shape",c1.shape)
    r1 = F.relu(c1)
    #print("r4.shape",r1.shape)
    p1 = self.pool(r1) 
    #print("p4.shape",p1.shape)

    #x = self.pool(F.relu(self.conv5(x)))
    c1 = self.conv5(p1)
    #print("c5.shape",c1.shape)
    r1 = F.relu(c1)
    #print("r5.shape",r1.shape)
    p1 = self.pool(r1) 
    #print("p5.shape",p1.shape)

    x = torch.flatten(p1, 1) # flatten all dimensions except batch
    #print('flatten', x.shape)
    
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x


cnn = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(cnn.parameters(),lr=0.001,momentum=0.9)

batch_size = 64
Dataset = iNatureDataset(trainingData,labels)
train_dl = DataLoader(Dataset, batch_size)

for epoch in range(1):
  running_loss = 0.0
  for i,data in enumerate(train_dl,0):
    inputs,labels = data
    #print('labels',labels)
    #labels = torch.tensor(labels)
    optimizer.zero_grad()
    outputs = cnn(inputs)

    loss = criterion(outputs,labels)
    loss.backward()
    optimizer.step()

    running_loss += loss.item()
    if i % 50 == 0:
      print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 50:.3f}')
      running_loss = 0.0

print('Finished Training')
