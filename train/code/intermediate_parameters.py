import pandas as pd
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
# from train.code.model import Net

import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation_1(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.activation_2(x)
        x = self.pool(x)

        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.fc1(x)
        x = self.activation_3(x)

        x = self.fc2(x)
        x = self.activation_4(x)
        x = self.fc3(x)
        return x


batch_size = 4

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)
activation = {}
sorted_node = {}
average = {}
ord_index_fc = {}


def get_index(df_avg):
    df_avg_list = []
    for i in df_avg[0]:
        df_avg_list.append(i)

    df_sort = df_avg_list
    df_sort = sorted(df_sort, key=float, reverse=True)
    out_index = []

    for ele in df_sort:
        index = df_avg_list.index(ele)
        out_index.append(index)
    ord_index = pd.DataFrame(out_index, columns=['Order'])
    return ord_index


def sort_nodes(activation, name):
    # print('Name: ', name)
    average[name] = torch.mean(activation[name], axis=0)
    l = pd.DataFrame(activation[name])
    l = l.transpose()
    m = pd.DataFrame(average[name])
    ord_index = get_index(m)
    n = pd.concat([l, m, ord_index], axis=1)
    n.to_csv(f'{name}_weights.csv')
    return ord_index


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
        if activation[name].dim() == 2:
            # print('2D array got in: ', name)
            ord_index_fc[name] = sort_nodes(activation, name)
    return hook


def main():
    model = Net()
    model.load_state_dict(torch.load('../../model/cifar_net.pth'))
    model.eval()
    # print(model)

    dataiter = iter(testloader)
    images, labels = next(dataiter)

    model.conv1.register_forward_hook(get_activation('conv1'))
    model.activation_1.register_forward_hook(get_activation('activation_1'))
    model.activation_2.register_forward_hook(get_activation('activation_2'))
    model.activation_3.register_forward_hook(get_activation('activation_3'))
    model.activation_4.register_forward_hook(get_activation('activation_4'))

    outputs = model(images)
    df = pd.DataFrame()

    for key in ord_index_fc.keys():
        df_temp = ord_index_fc[key]
        df_temp.rename(columns={'Order': key}, inplace=True)
        df = pd.concat([df, df_temp], axis=1)
        print(df.shape)
    df.to_csv('Order_out.csv')

if __name__ == '__main__':
    main()
