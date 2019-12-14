import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as utils

import matplotlib.pyplot as plt


FEATURE_LEN = 70


###############################################
## Main
###############################################

def main():
    infile = "data.csv"
    data = pd.read_csv(infile)

    names = list(data)[1:]

    train = []
    dev = []
    test = []

    for i in range(len(names)):
        if i % 5 < 3:
            train.append(names[i])
        elif i % 5 < 4:
            dev.append(names[i])
        else:
            test.append(names[i])

    print("Train split:", train)
    print("Dev split:", dev)
    print("Test split:", test)

    train_x = []
    train_y = []

    dev_x = []
    dev_y = []

    for train_name in train:
        v = data[train_name].values[:-2]

        v_mean = np.mean(v[:100])
        v_diff = v[99] - v[0]
        v = (v - v_mean) / v_diff

        for i in range(100, len(v)):
            train_x.append(v[(i-FEATURE_LEN):i])
            train_y.append(v[i])


    for dev_name in dev:
        v = data[dev_name].values[:-2]

        v_mean = np.mean(v[:100])
        v_diff = v[99] - v[0]
        v = (v - v_mean) / v_diff

        for i in range(100, len(v)):
            dev_x.append(v[(i-FEATURE_LEN):i])
            dev_y.append(v[i])


    torch_tx = torch.stack([torch.Tensor(a).float() for a in train_x])
    torch_ty = torch.Tensor(train_y)
    train_set = utils.TensorDataset(torch_tx, torch_ty)

    torch_dx = torch.stack([torch.Tensor(a).float() for a in dev_x])
    torch_dy = torch.Tensor(dev_y)
    dev_set = utils.TensorDataset(torch_dx, torch_dy)

    train_loader = utils.DataLoader(train_set, batch_size=1, shuffle=True)
    dev_loader = utils.DataLoader(dev_set, batch_size=1, shuffle=False)



    torch.set_default_tensor_type(torch.FloatTensor)

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(FEATURE_LEN, 64)
            self.fc2 = nn.Linear(64, 16)
            self.fc3 = nn.Linear(16, 1)

            torch.nn.init.xavier_uniform(self.fc1.weight)
            torch.nn.init.xavier_uniform(self.fc2.weight)
            torch.nn.init.xavier_uniform(self.fc3.weight)

        def forward(self, x, training=True):
            x = F.relu(self.fc1(x))
            x = F.dropout(x, 0.7, training=training)
            x = F.relu(self.fc2(x))
            x = F.dropout(x, 0.7, training=training)

            return self.fc3(x)

        def predict(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))

            return self.fc3(x)

    net = Net()
    net = net.float()

    ## Drive to device
    device = torch.device("cpu")
    net.to(device)

    ## Set net mode
    net = net.train()
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-5)

    for epoch in range(15):  # loop over the dataset multiple times
        running_loss = 0.0

        for i, dat in enumerate(train_loader, 0):
            # get the inputs
            inputs, labels = dat

            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)[0]

            # print(inputs, " -- ", labels, " -- ", outputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print('[Epoch: %d] Current loss: %.10f' % (epoch + 1, running_loss))
        print('-- Average loss: %.10f' % (running_loss / len(train_loader)))

    print('Finished Training')

    net = net.eval()

    loss = 0
    total = 0
    with torch.no_grad():
        for input, labels in dev_loader:
            outputs = net(input)
            loss += criterion(outputs, labels).sum().item()
            total += 1

    print('Average loss of the network on test data: %.2f' % (
        loss / total))

    loss = []

    for dev_name in dev:
        dev_v = data[dev[2]].values[:-2]
        dev_input = dev_v[:100]

        v = dev_input

        v_mean = np.mean(v[:100])
        v_diff = v[99] - v[0]
        dev_input = (v - v_mean) / v_diff


        for i in range(48):
            pred_input = torch.Tensor(dev_input[-FEATURE_LEN:]).float()
            pred_v= net(pred_input)[0].item()
            dev_input = np.append(dev_input, pred_v)

        x = data["epoch"].values[:-2]
        dev_input = dev_input * v_diff + v_mean

        loss.append(np.sum(np.abs(dev_input[-3:] - dev_v[-3:])))

        # print("Loss: ", loss[-1])

    loss = np.array(loss)
    print("Max loss:", np.max(loss))
    print("Min loss:", np.min(loss))
    print("Average loss:", np.mean(loss))

    dev_v = data[dev[2]].values[:-2]
    dev_input = dev_v[:100]

    v = dev_input

    v_mean = np.mean(v[:100])
    v_diff = v[99] - v[0]
    dev_input = (v - v_mean) / v_diff


    for i in range(48):
        pred_input = torch.Tensor(dev_input[-FEATURE_LEN:]).float()
        pred_v= net(pred_input)[0].item()
        dev_input = np.append(dev_input, pred_v)

    x = data["epoch"].values[:-2]
    dev_input = dev_input * v_diff + v_mean

    plt.axvline(x=100, linestyle="--", color="grey")
    plt.plot(x[95:], dev_input[95:], label="prediction", linestyle="--", color="green", alpha=0.7)
    plt.plot(x, dev_v, label="target", color="orange", alpha=0.5)

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Sliding Window Prediction")

    plt.legend()
    plt.savefig("nn_pred.png")
    plt.show()



if __name__=='__main__':
    main()
