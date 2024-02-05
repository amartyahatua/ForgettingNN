# import torch
# import torchvision
# import torchvision.transforms as transforms
# import matplotlib.pyplot as plt
# import numpy as np
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import pandas as pd
#
# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#
# batch_size = 4
#
# trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                         download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
#                                           shuffle=True, num_workers=2)
#
# testset = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                        download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
#                                          shuffle=False, num_workers=2)
#
# classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#
#
# class MyLayer(nn.Module):
#     """
#     This is a user defined neural network layer
#     param: int input size
#     param: tensor Input * random weight
#     """
#
#     def __init__(self, in_size):
#         super().__init__()
#         self.weight = None
#         self.size = in_size
#
#     def forward(self, x, n_epochs, layer):
#         df = pd.read_csv('Order_out_old.csv')
#         rank = torch.tensor(df[layer].values[0:x.shape[1]])
#         #rank = rank.float()
#         rank = [i  for i in range(self.size) ]
#         rank = torch.tensor(rank)
#         self.weight = np.exp(-n_epochs / rank)
#         return x * self.weight
#
#
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#
#         # Convolutional layer with 64 filters, 3x3 kernel, and ReLU activation
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
#
#         # MaxPooling2D layer with 2x2 pool size and Dropout of 0.25 probability
#         self.pool = nn.MaxPool2d(2, 2)
#         self.dropout1 = nn.Dropout2d(p=0.25)
#
#         # Fully Connected layer with 128 units, tanh activation, and Dropout of 0.5 probability
#         self.fc1 = nn.Linear(64 * 16 * 16, 128)  # Adjust the input size based on your input dimensions
#         self.tanh = nn.Tanh()
#         self.dropout2 = nn.Dropout(p=0.5)
#
#         # Fully Connected layer with 10 units and softmax activation
#         self.fc2 = nn.Linear(128, 10)
#         self.myLayer_1 = MyLayer(128)
#         self.myLayer_2 = MyLayer(10)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = nn.functional.relu(x)
#         x = self.pool(x)
#         x = self.dropout1(x)
#
#         x = x.view(-1, 64 * 16 * 16)  # Adjust the view size based on your input dimensions
#         x = self.fc1(x)
#         self.fc1.weight.data = nn.Parameter(torch.tensor(np.exp(-self.fc1.weight.data.detach().numpy())))
#         print(self.fc1.weight)
#         x = self.tanh(x)
#         x = self.dropout2(x)
#
#         x = self.fc2(x)
#         #self.fc2.weight = nn.Parameter(torch.tensor(np.exp(-self.fc2.weight.detach().numpy() / self.epoch)))
#         x = nn.functional.softmax(x, dim=1)
#
#         return x
#
# # class Net(nn.Module):
# #     def __init__(self):
# #         super().__init__()
# #         self.conv1 = nn.Conv2d(3, 6, 5)
# #         self.pool = nn.MaxPool2d(2, 2)
# #         self.conv2 = nn.Conv2d(6, 16, 5)
# #         self.activation_1 = nn.ReLU()
# #         self.activation_2 = nn.ReLU()
# #         self.activation_3 = nn.ReLU()
# #         self.activation_4 = nn.ReLU()
# #         self.fc1 = nn.Linear(16 * 5 * 5, 120)
# #         self.fc2 = nn.Linear(120, 84)
# #         self.fc3 = nn.Linear(84, 10)
# #         self.myLayer_3 = MyLayer(120)
# #         self.myLayer_4 = MyLayer(84)
# #
# #     def forward(self, x):
# #         x = self.conv1(x)
# #         x = self.activation_1(x)
# #         x = self.pool(x)
# #
# #         x = self.conv2(x)
# #         x = self.activation_2(x)
# #         x = self.pool(x)
# #
# #         x = torch.flatten(x, 1)
# #         x = self.fc1(x)
# #         x = self.activation_3(x)
# #         x = self.myLayer_3(x, self.epoch, 'activation_3')
# #
# #         x = self.fc2(x)
# #         x = self.activation_4(x)
# #         x = self.myLayer_4(x, self.epoch, 'activation_4')
# #         x = self.fc3(x)
# #
# #         return x
#
#
# net = Net()
#
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
#
#
# def train():
#
#     for turn in range(1):
#         for epoch in range(20):  # loop over the dataset multiple times
#             print("Epoch", epoch)
#             running_loss = 0.0
#             for i, data in enumerate(trainloader, 0):
#                 # get the inputs; data is a list of [inputs, labels]
#                 inputs, labels = data
#
#                 # zero the parameter gradients
#                 optimizer.zero_grad()
#
#                 # forward + backward + optimize
#                 net.epoch = epoch + 1
#                 outputs = net(inputs)
#                 loss = criterion(outputs, labels)
#                 loss.backward()
#                 optimizer.step()
#
#                 # print statistics
#                 running_loss += loss.item()
#                 if i % 2000 == 1999:  # print every 2000 mini-batches
#                     print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
#                     running_loss = 0.0
#         PATH = f'../../model/cifar_net_{turn}.pth'
#         torch.save(net.state_dict(), PATH)
#         print('Finished Training')
#
#         correct = 0
#         total = 0
#         # since we're not training, we don't need to calculate the gradients for our outputs
#         with torch.no_grad():
#             for data in testloader:
#                 images, labels = data
#                 # calculate outputs by running images through the network
#                 outputs = net(images)
#                 # the class with the highest energy is what we choose as prediction
#                 _, predicted = torch.max(outputs.data, 1)
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()
#
#         print(f'Accuracy of the network on the 10000 test images in Turn: {turn}: {100 * correct // total} %')
#
#
# if __name__ == '__main__':
#     train()

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
# from train.code.resnet import ResNet

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.res1 = nn.Sequential(nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        ), nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.res2 = nn.Sequential(nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        ), nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))
        )

        self.classifier = nn.Sequential(
            nn.MaxPool2d(4),
            nn.Flatten(),
            nn.Linear(512, 10)

        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.res1(x) + x
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.res2(x) + x
        x = self.classifier(x)
        x = nn.Parameter(torch.tensor(np.exp(-x.detach().numpy())))

        return x

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class MyLayer(nn.Module):
    """
    This is a user defined neural network layer
    param: int input size
    param: tensor Input * random weight
    """

    def __init__(self, in_size):
        super().__init__()
        self.weight = None
        self.size = in_size

    def forward(self, x, n_epochs, layer):
        df = pd.read_csv('Order_out_old.csv')
        # rank = torch.tensor(df[layer].values[0:x.shape[1]])
        # rank = rank.float()
        rank = (df[layer].values[0:x.shape[1]])
        self.weight = nn.Parameter(torch.tensor(np.exp(-n_epochs / (rank+1))))
        # print('Weight shape: ',self.weight.shape)
        x = torch.tensor(np.exp(-x.detach().numpy()*rank))
        return x.float()*self.weight.float()


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.activation_1 = nn.ReLU()
        self.activation_2 = nn.ReLU()
        self.activation_3 = nn.ReLU()
        self.activation_4 = nn.ReLU()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.myLayer_3 = MyLayer(120)
        self.myLayer_4 = MyLayer(84)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation_1(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.activation_2(x)
        x = self.pool(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.activation_3(x)

        if(self.turn == 'unlearning'):
            #x = self.myLayer_3(x, self.epoch, 'activation_3')
            x = torch.tensor(np.exp(-x.detach().numpy()))

        x = self.fc2(x)
        x = self.activation_4(x)

        if(self.turn == 'unlearning'):
            #x = self.myLayer_4(x, self.epoch, 'activation_4')
            x = torch.tensor(np.exp(-x.detach().numpy()))
        x = self.fc3(x)

        return x


net = Net()
# net = ResNet()


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

def test(type):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the network on the 10000 test images on {type}: {100 * correct // total} %')

def train():
    for turn in range(10):
        print(f'------------------------Turn = {turn}-----------------------------')
        for epoch in range(10):  # loop over the dataset multiple times
            print("Learning epoch", epoch)
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                net.epoch = epoch + 1
                net.turn = 'training'
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0
            test('Learning')

        for epoch in range(5):  # loop over the dataset multiple times
            print("Unlearning epoch: ", epoch)
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                net.epoch = epoch + 1
                net.turn = 'unlearning'
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0
            test('Unlearning')


if __name__ == '__main__':
    train()
