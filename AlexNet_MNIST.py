import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

batch_size = 128
learning_rate = 0.0002
num_epoch = 20

is_gpu = True

mnist_train = dset.MNIST("./", train=True, transform=transforms.ToTensor(), target_transform = None, download = True)
mnist_test = dset.MNIST("./", train=False, transform=transforms.ToTensor(), target_transform = None, download = True)

print(mnist_train.__getitem__(0)[0].size(), mnist_train.__len__())
mnist_test.__getitem__(0)[0].size(), mnist_test.__len__()

train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle = False)

class AlexNet(nn.Module) :
    def __init__(self):
        super(AlexNet, self).__init__()
        self.layer = nn.Sequential(
            #1st
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            
            #2nd
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),

            #3rd
            nn.Conv2d(128, 256, 3, 1, 1),
            #nn.BatchNorm2d(384),
            nn.ReLU(),

            #4th
            nn.Conv2d(256, 512, 3, 1, 1),
            #nn.BatchNorm2d(384),
            nn.ReLU(),

            #5th
            nn.Conv2d(512, 512, 3, 1, 0),
            #nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
        )
        #6th
        self.flatten6 = nn.Flatten()
        self.fc_layer6 = nn.Sequential(
            nn.LazyLinear(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            #nn.Linear(4096, 10)
        )
        #7th
        self.flatten7 = nn.Flatten()
        self.fc_layer7 = nn.Sequential(
            nn.LazyLinear(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            #nn.Linear(400, 10)
        )
        self.fc_layer_out = nn.Linear(512, 28)

        
    def forward(self, x) :
        out = self.layer(x)
        #out = out.view(batch_size, -1)
        out = self.flatten6(out)
        out = self.fc_layer6(out)
        out = self.flatten7(out)
        out = self.fc_layer7(out)
        out = self.fc_layer_out(out)
        return out
        
model = AlexNet()

if torch.cuda.is_available() :
    model = model.cuda()

loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

num_epoch = 20 #원랜 20
trial = num_epoch*len(train_loader) #설정 안 하면 num_epoch * len(train_laoder)(469) 만큼..
print(num_epoch, len(train_loader), trial)
for i in range(num_epoch) :
    print()
    print(i + 1, '번째 epoch')
    stp = 0
    for j, [image, label] in enumerate(train_loader) :
        optimizer.zero_grad()
        
        if torch.cuda.is_available() :
            x = Variable(image).cuda()
            y = Variable(label).cuda()
        else :
            x = Variable(image)
            y = Variable(label)
            
        output = model.forward(x)
        loss = loss_func(output, y)
        loss.backward()
        optimizer.step()
        
        if j % 1000 == 0 :
            print(stp + 1, '번째 loss : ', loss)
        stp += 1
        if stp == trial :
            break
print()
print('end')

param_list = []
param_list = list(model.parameters())
print(param_list)

correct = 0
total = 0

for image, label in test_loader :
    if torch.cuda.is_available() :
        x = Variable(image, volatile = True).cuda()
        y = Variable(label).cuda()

    else :
        x = Variable(image, volatile = True)
        y = Variable(label)
    
    output = model.forward(x)
    _, output_index = torch.max(output, 1)
    
    total += label.size(0)
    correct += (output_index == y).sum().float()

print("Accuracy of Test Data: {}".format(100*correct/total))