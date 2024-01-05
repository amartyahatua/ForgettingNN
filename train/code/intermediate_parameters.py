import pandas as pd
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from model import Net

transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
batch_size = 4
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)
activation = {}
sorted_node = {}
def sort_nodes(activation, name):
    print('Name: ',name)
    sort_index = np.argsort(activation[name] )
    sorted_node[name] = torch.mean(sort_index.float(), axis=0)
    l = pd.DataFrame(activation[name])
    l = l.transpose()
    m = pd.DataFrame(sorted_node[name])
    n = pd.concat([l,m], axis=1)
    n.to_csv(f'{name}_weights.csv')

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
        if(activation[name].dim() == 2):
            print('2D array got in: ', name)
            sort_nodes(activation, name)
    return hook

def main():
    model = Net()
    model.load_state_dict(torch.load('../../model/cifar_net.pth'))
    model.eval()
    print(model)

    dataiter = iter(testloader)
    images, labels = next(dataiter)

    model.conv1.register_forward_hook(get_activation('conv1'))
    model.activation_1.register_forward_hook(get_activation('activation_1'))
    model.activation_2.register_forward_hook(get_activation('activation_2'))
    model.activation_3.register_forward_hook(get_activation('activation_3'))
    model.activation_4.register_forward_hook(get_activation('activation_4'))

    outputs = model(images)
    print(activation.keys())
    for key in activation.keys():
        print(f'{key} shape: {activation[key].shape}')

if __name__ == '__main__':
    main()
