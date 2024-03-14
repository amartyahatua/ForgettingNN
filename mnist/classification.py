import os
import argparse
import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from matplotlib import pyplot as plt
from mia_score import calculate_mia
import random

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
torch.autograd.set_detect_anomaly(True)


def get_index(df_avg):
    L = []
    for val in df_avg.values.tolist():
        L.extend(val)
    x = tuple(k[1] for k in sorted((x[1], j) for j, x in enumerate(
        sorted((x, i) for i, x in enumerate(L)))))
    ord_index = [max(x) - i for i in list(x)]
    return ord_index


def node_order(weights):
    average = torch.mean(weights, axis=0)
    new_average = pd.DataFrame(average.cpu().detach().numpy())
    ord_index = get_index(new_average)
    return ord_index


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.dropout1 = None
        self.dropout2 = None
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        if self.type == 'unlearning':
            prev_drp = 0
            try:
                self.dropout1 = nn.Dropout(0.25 - (0.072 * self.turn) + (0.05 * (self.epoch - 1)))
                self.dropout2 = nn.Dropout(0.50 - (0.072 * self.turn) + (0.05 * (self.epoch - 1)))
                prev_drp = (0.25 - (0.072 * self.turn) + (0.05 * (self.epoch - 1)))
            except:
                self.dropout1 = nn.Dropout(prev_drp)
                self.dropout2 = nn.Dropout(prev_drp)
        elif self.type == 'learning':
            self.dropout1 = nn.Dropout(0.25 - (0.01 * self.turn))
            self.dropout2 = nn.Dropout(0.50 - (0.01 * self.turn))

        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)

        if self.type == 'unlearning':
            # With ordered numbers
            rank = torch.tensor([i for i in range(x.shape[1])])
            rank = rank.to(self.device)

            # # With ordered nodes
            # rank = node_order(x)
            # rank = torch.tensor(rank)
            # rank = rank.to(self.device)
            #
            # # With top 30 nodes
            # rank = node_order(rank)
            # rank = [ind if 30 <= ind else 0 for ind in rank]
            #
            # # Random node
            # random_numbers = random.sample(range(0, x.shape[1] - 1), random.randint(0, x.shape[1] - 1))
            # rank = node_order(x)
            # for rn in random_numbers:
            #     try:
            #         rank[rn] = 1
            #     except:
            #         continue
            #     rank = torch.tensor(rank)
            #     rank = rank.to(self.device)
            x = x * torch.exp(-(self.forget_strength / rank))

        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        if self.type == 'unlearning':
            # With ordered numbers
            rank = torch.tensor([i for i in range(x.shape[1])])
            rank = rank.to(self.device)

            # # With ordered nodes
            # rank = node_order(x)
            # rank = torch.tensor(rank)
            # rank = rank.to(self.device)
            #
            # # With top 30 nodes
            # rank = node_order(rank)
            # rank = [ind if 30 <= ind else 0 for ind in rank]
            #
            # # Random node
            # random_numbers = random.sample(range(0, x.shape[1] - 1), random.randint(0, x.shape[1] - 1))
            # rank = node_order(x)
            # for rn in random_numbers:
            #     try:
            #         rank[rn] = 1
            #     except:
            #         continue
            #     rank = torch.tensor(rank)
            #     rank = rank.to(self.device)
            x = x * torch.exp(-(self.forget_strength / rank))

        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch, type, turn):
    model.train()
    model.type = type
    model.forget_strength = epoch
    model.turn = turn
    model.rank = 3
    model.last_layer_forget = 0
    model.device = device

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


accuracy = []
accuracy_num = []
mia_score_list = []

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    print(f'Learning type: {type}; droupout 1: {model.dropout1};  droupout 2: {model.dropout2}')
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    accuracy.append(correct / len(test_loader.dataset))
    accuracy_num.append(correct)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.001, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.07, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    # parser.add_argument('--no-mps', action='store_true', default=False,
    #                     help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=500, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    # use_mps = not args.no_mps and torch.backends.mps.is_available()
    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    # elif use_mps:
    #     device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                              transform=transform)
    dataset2, dataset3 = torch.utils.data.random_split(dataset1, [50000, 10000], generator=torch.Generator().manual_seed(42))

    dataset4 = datasets.MNIST('../data', train=False,
                              transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    forget_loader = torch.utils.data.DataLoader(dataset3, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset4, **test_kwargs)

    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for turn in range(5):
        print(f'------------------------Turn = {turn}-----------------------------')
        print(f'------------------------Learning-----------------------------')
        for epoch in range(1, args.epochs + 2):
            train(args, model, device, train_loader, optimizer, epoch, 'learning', turn)
            test(model, device, test_loader)
            scheduler.step()
            mia_score_list.append(calculate_mia(model, test_loader, forget_loader))
        print(f'------------------------Unlearning-----------------------------')
        for epoch in range(1, args.epochs + 5):
            train(args, model, device, train_loader, optimizer, epoch, 'unlearning', turn)
            test(model, device, test_loader)
            scheduler.step()
            mia_score_list.append(calculate_mia(model, test_loader, forget_loader))

    if args.save_model:
        torch.save(model.state_dict(), "model/mnist_cnn.pt")


if __name__ == '__main__':
    main()
    print(accuracy)
    accuracy = pd.DataFrame(accuracy)
    mia_score_df = pd.DataFrame(mia_score_list)
    accuracy.to_csv('Result_Rank_in_order_all_epoch_all_layer.csv', index=False)
    mia_score_df.to_csv('MIA_Rank_in_order_all_epoch_all_layer.csv', index=False)
    plt.plot(accuracy)
    plt.savefig('plots/unlearning_plot_rank_in_order_all_epoch_all_layer.png')

    plt.plot(accuracy_num)
    plt.savefig('plots/unlearning_rank_in_order_all_epoch_all_layer.png')
