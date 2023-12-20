import torch
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
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def main():
    model = Net()
    model.load_state_dict(torch.load('../../model/cifar_net.pth'))
    model.eval()

    print(model)
    activation = {}

    dataiter = iter(testloader)
    images, labels = next(dataiter)

    model.conv1.register_forward_hook(lambda m, input, output: print(output.shape))
    model.pool.register_forward_hook(lambda m, input, output: print(output.shape))
    model.conv2.register_forward_hook(lambda m, input, output: print(output.shape))
    model.fc1.register_forward_hook(lambda m, input, output: print(output.shape))
    model.fc2.register_forward_hook(lambda m, input, output: print(output.shape))
    model.fc3.register_forward_hook(lambda m, input, output: print(output.shape))
    outputs = model(images)

if __name__ == '__main__':
    main()
