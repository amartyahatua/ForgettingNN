"""
The script demonstrates a simple example of using ART with PyTorch. The example train a small model on the MNIST dataset
and creates adversarial examples using the Fast Gradient Sign Method. Here we use the ART classifier to train the model,
it would also be possible to provide a pretrained model to the ART classifier.
The parameters are chosen for reduced computational requirements of the script and not optimised for accuracy.
"""
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch
import pandas as pd
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier
# from art.utils import load_mnist
from plots import create_plot
import random
from art.utils import load_mnist
from sklearn.model_selection import train_test_split

# Step 0: Define the neural network model, return logits instead of activation in forward method


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.dropout1 = None
        self.dropout2 = None
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def get_index(self, df_avg):
        L = []
        for val in df_avg.values.tolist():
            L.extend(val)
        x = tuple(k[1] for k in sorted((x[1], j) for j, x in enumerate(
            sorted((x, i) for i, x in enumerate(L)))))
        ord_index = [max(x) - i for i in list(x)]
        return ord_index

    def node_order(self, weights):
        average = torch.mean(weights, axis=0)
        new_average = pd.DataFrame(average.cpu().detach().numpy())
        ord_index = self.get_index(new_average)
        return ord_index

    def forward(self, x):
        if self.type == 'unlearning':
            prev_drp = 0
            try:
                self.dropout1 = nn.Dropout(0.65 - (0.02 * (self.turn-1)) + (0.05 * (self.epoch - 1)))
                self.dropout2 = nn.Dropout(0.75 - (0.02 * (self.turn-1)) + (0.05 * (self.epoch - 1)))
            except:
                self.dropout1 = nn.Dropout(0.65)
                self.dropout2 = nn.Dropout(0.75)
        elif self.type == 'learning':
            self.dropout1 = nn.Dropout(0.25)
            self.dropout2 = nn.Dropout(0.50)

        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)

        # if self.type == 'unlearning':
            # With ordered numbers
            # rank = torch.tensor([i for i in range(x.shape[1])])
            # rank = rank.to(self.device)

            # With ordered nodes
            # rank = self.node_order(x)
            # rank = torch.tensor(rank)
            # rank = rank.to(self.device)

            # With to 30 nodes
            # rank = self.node_order(x)
            # rank = [ind if 30 <= ind else 0 for ind in rank]
            # rank = torch.tensor(rank)
            # rank = rank.to(self.device)

            # Random node
            # random_numbers = random.sample(range(0, x.shape[1] - 1), random.randint(0, x.shape[1] - 1))
            # rank = self.node_order(x)
            # for rn in random_numbers:
            #     self.dropout2 = nn.Dropout(rn / 1000)
            #     try:
            #         rank[rn] = 1
            #     except:
            #         continue
            # rank = torch.tensor(rank)
            # rank = rank.to(self.device)
            # x = x * torch.exp(-(self.epoch / rank))



        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        if self.type == 'unlearning':
            # With ordered numbers
            # rank = torch.tensor([i for i in range(x.shape[1])])
            # rank = rank.to(self.device)

            # With ordered nodes
            rank = self.node_order(x)
            rank = torch.tensor(rank)
            rank = rank.to(self.device)

            # With to 30 nodes
            # rank = self.node_order(x)
            # rank = [ind if 30 <= ind else 0 for ind in rank]
            # rank = torch.tensor(rank)
            # rank = rank.to(self.device)

            # Random node
            # random_numbers = random.sample(range(0, x.shape[1] - 1), random.randint(0, x.shape[1] - 1))
            # rank = self.node_order(x)
            # for rn in random_numbers:
            #     try:
            #         rank[rn] = 1000
            #     except:
            #         continue
            # rank = torch.tensor(rank)
            # rank = rank.to(self.device)
            x = x * torch.exp(-(self.epoch / rank))

        output = F.log_softmax(x, dim=1)
        return output


# Step 1: Load the MNIST dataset

(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()
x_train = np.transpose(x_train, (0, 3, 1, 2)).astype(np.float32)
x_test = np.transpose(x_test, (0, 3, 1, 2)).astype(np.float32)


# Step 2: Create the model
model = Net()

# Step 2a: Define the loss function and the optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


# Step 3: Create the ART classifier


def train(model, x_train, y_train, epoch, type, device):
    classifier = PyTorchClassifier(
        model=model,
        clip_values=(min_pixel_value, max_pixel_value),
        loss=criterion,
        optimizer=optimizer,
        input_shape=(1, 28, 28),
        nb_classes=10
    )

    model.train()
    model.type = type
    model.epoch = epoch
    model.device = device
    optimizer.zero_grad()
    optimizer.step()
    classifier.fit(x_train, y_train, batch_size=64, nb_epochs=epoch)
    return classifier



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
accuracy_list = []
mia_score_list = []

print(f'------------------------Learning-----------------------------')
# Step 4: Train the ART classifier
for epoch in range(1, 31):
    classifier = train(model, x_train, y_train, epoch, 'learning', device)
    # Step 5: Evaluate the ART classifier on benign test examples
    predictions = classifier.predict(x_test)
    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
    print("Accuracy on benign test examples: {:0.2f}%".format(accuracy * 100))
    accuracy_list.append(accuracy)

    # Step 6: Generate adversarial test examples
    attack = FastGradientMethod(estimator=classifier, eps=0.2)
    x_test_adv = attack.generate(x=x_test)

    # Step 7: Evaluate the ART classifier on adversarial test examples
    predictions = classifier.predict(x_test_adv)
    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
    mia_score_list.append(accuracy)
    print("Accuracy on adversarial test examples: {:0.2f}%".format(accuracy * 100))

print(f'------------------------Unlearning-----------------------------')
for epoch in range(1, 31):
    X_retain, X_forget, y_retain, y_forget = train_test_split(x_train, y_train, random_state=104, test_size=0.25, shuffle=True)

    classifier = train(model, X_retain, y_retain, epoch, 'learning', device)
    # Step 5: Evaluate the ART classifier on benign test examples
    predictions = classifier.predict(X_forget)
    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_forget, axis=1)) / len(y_forget)
    print("Accuracy on benign test examples: {:0.2f}%".format(accuracy * 100))
    accuracy_list.append(accuracy)

    # Step 6: Generate adversarial test examples
    attack = FastGradientMethod(estimator=classifier, eps=0.2)
    x_test_adv = attack.generate(x=X_forget)

    # Step 7: Evaluate the ART classifier on adversarial test examples
    predictions = classifier.predict(x_test_adv)
    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_forget, axis=1)) / len(y_forget)
    mia_score_list.append(accuracy)
    print("Accuracy on adversarial test examples: {:0.2f}%".format(accuracy * 100))

df_accuracy = pd.DataFrame(accuracy_list)
mia_score_df = pd.DataFrame(mia_score_list)
df_accuracy.to_csv('result/fashion/Baseline_accuracy.csv', index=False)
mia_score_df.to_csv('result/fashion/Baseline_mia.csv', index=False)
