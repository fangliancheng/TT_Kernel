import torch
import numpy as np
import pandas as pd
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn as nn
from vLinear import Grnn
from vF import comp

torch.backends.cudnn.deterministic = True
torch.manual_seed(9)

dtype = torch.float
# device = torch.device("cpu")
device = torch.device("cuda:0")  # Uncomment this to run on GPU

# Load data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
bsize = 250
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=bsize, shuffle=True, num_workers=2)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=bsize, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# neurons_A: emb_dim=32  num_hidden:R=64     output_size: 10       input_size: raw x length=49
class TTRNN(nn.Module):
# output_size only relative to the last FC layer, in mid GRNN layer, no output_size need to be mannuly specified
    def __int__(self, input_size, num_hidden, num_neurons_A, output_size, gen):
        super(TTRNN, self).__init__()
        #rnn_i is an object of class Grnn

        self.input_size = None
        self.output_size = None
        self.num_neurons_A = 49
        self.num_hidden = 64
        self.gen = 'relu'

        self.rnn = Grnn(input_size, num_neurons_A, num_hidden, gen)
        self.last = nn.Linear(input_size, num_output)

    def forward(self, input, hidden):
        output, hidden = self.rnn(input, hidden)
        return output, hidden

    def initHidden1(self):
        return torch.zeros(1, self.num_hidden)

    def inithidden2(self):
        return torch.ones(1, self.num_hidden)


# helper function

def rgb(images):
    image_np = images.numpy()
    image_input = np.zeros(((bsize, 28, 28)))
    for i in range(0, 28):
        for j in range(0, 28):
            image_input[:, i, j] = image_np[:, 0, i, j]
    return image_input


def one_hot(label):
    batch_size = bsize
    nb_digits = 10
    # Dummy input that HAS to be 2D for the scatter (you can use view(-1,1) if needed)
    y = label.view(-1, 1).type(torch.LongTensor).to(device)
    # One hot encoding buffer that you create out of the loop and just keep reusing
    y_onehot = torch.FloatTensor(batch_size, nb_digits).to(device)

    # In your for loop
    y_onehot.zero_()
    y_onehot.scatter_(1, y, 1)
    y_onehot.to(device)
    return y_onehot


# images: batch_size * 28 * 28 for Minst
def feature_map(bsize, images):
    num_split = 4
    piece_length = 49
    temp = np.zeros(((bsize, piece_length, 16)))
    upper1 = np.split(np.split(images, num_split, axis=1)[0], num_split, axis=2)
    upper2 = np.split(np.split(images, num_split, axis=1)[1], num_split, axis=2)
    upper3 = np.split(np.split(images, num_split, axis=1)[2], num_split, axis=2)
    upper4 = np.split(np.split(images, num_split, axis=1)[3], num_split, axis=2)

    for i in range(0, bsize):
        for j in range(0, 4):
            temp[i, :, j] = (upper1[j])[i, :, :].flatten()
        for j in range(4, 8):
            temp[i, :, j] = (upper2[j-4])[i, :, :].flatten()
        for j in range(8, 12):
            temp[i, :, j] = (upper3[j-8])[i, :, :].flatten()
        for j in range(12, 16):
            temp[i, :, j] = (upper4[j-12])[i, :, :].flatten()
    return temp


def test():
    # Test part
    correct = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            images, labels = data
            images.to(device)
            labels.to(device)

            output, hidden = tt_rnn(input, hidden)
            outputs = g_inner(weights_CP, images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.to(device) == labels.to(device)).sum().item()
    return 100.0 * (correct) / total


def train(epoch):
    num_timesteps = 16
    num_output = 10
    input_size = 49
    num_hidden = 64
    num_neurons_A = 32
    embed_dim = 32
    num_label = 10
    patch_size = 7
    generalization = "relu"

    # instantiate the class
    # self, input_size, num_hidden, num_neurons_A, output_size, gen
    tt_rnn = TTRNN()
    tt_rnn.input_size = 49
    tt_rnn.output_size = 10
    tt_rnn.num_hidden = 64
    tt_rnn.num_neurons_A = 32
    tt_rnn.gen = "relu"

    learning_rate = 1e-4
    print(learning_rate, tt_rnn.gen)

    #if tt_rnn.gen == "relu":
    hidden = tt_rnn.initHidden1()
    #if tt_rnn.gen == "prod":
    #    hidden = tt_rnn.inithidden2()
    #else:
    #    print(tt_rnn.gen)
    #    print("error activation input")

    tt_rnn.zero_grad()

    bsize = 250
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        print(inputs.size())
        #inputs.to(device)

        if inputs.size()[0] != bsize:
            continue
        inputs = torch.from_numpy(rgb(inputs))
        input = feature_map(bsize, inputs)
        hidden = hidden
        y = labels.float()
        y = one_hot(y)
        #particular: hidden = output in each time step
        for i in range(0, num_timesteps):
            output, hidden = tt_rnn(input[:, :, i], hidden)
        output = tt_rnn.last(num_hidden, num_output)
        y_predict = torch.maximum(output, dim=1)
        loss = (y_predict - y).pow(2).sum()
        loss.backward()

        for p in tt_rnn.parameters():
            p.data.add(-learning_rate, p.grad.data)

        return output, loss.item


epoch_num = 300
for epoch in range(1, epoch_num):
    train(epoch)
print("training finished")



