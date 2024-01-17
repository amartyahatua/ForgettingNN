import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class MyLayer(nn.Module):
    """
    This is a user defined neural network layer
    param: int input size
    param: tensor Input * random weight
    """
    def __init__(self, in_size, n_epochs, rank):
        super().__init__()
        self.weight = None
        self.size = in_size
        self.epoch = n_epochs
        self.rank = rank

    def forward(self, x):
        self.weight = np.exp(-self.epoch / self.r)
        return x * self.weight


# load the dataset, split into input (X) and output (y) variables
dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')
X = dataset[:, 0:8]
y = dataset[:, 8]

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# define the model
model = nn.Sequential(
    nn.Linear(8, 12),
    MyLayer(12),
    nn.ReLU(),
    nn.Linear(12, 8),
    MyLayer(8),
    nn.ReLU(),
    nn.Linear(8, 1),
    MyLayer(1),
    nn.Sigmoid(),
)
print(model)

# train the model
loss_fn = nn.BCELoss()  # binary cross entropy
optimizer = optim.Adam(model.parameters(), lr=0.001)

n_epochs = 10
batch_size = 10


def run_nn():
    for epoch in range(n_epochs):
        for i in range(0, len(X), batch_size):
            Xbatch = X[i:i + batch_size]
            y_pred = model(Xbatch)
            ybatch = y[i:i + batch_size]
            loss = loss_fn(y_pred, ybatch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # print(f'Finished epoch {epoch}, latest loss {loss}')

    # compute accuracy (no_grad is optional)
    with torch.no_grad():
        y_pred = model(X)
    accuracy = (y_pred.round() == y).float().mean()
    print(f"Accuracy {accuracy}")


if __name__ == '__main__':
    run_nn()
