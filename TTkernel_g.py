import torch
import numpy as np
import pandas as pd
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn as nn
from torch.nn.parameter import parameter
from .. import functional as F
from .. import init
from .module import Module
from ..._jit_internal import weak_module, weak_script_method

torch.backends.cudnn.deterministic = True
torch.manual_seed(9)

dtype = torch.float
device = torch.device("cpu")
device = torch.device("cuda:0") # Uncomment this to run on GPU

# Load data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=bsize, shuffle=True, num_workers=2)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=bsize, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# batch size
bsize = 250

num_timesteps = 16
num_hidden = 64
num_neurons_A = 49
embed_dim = 32
num_label = 10
patch_size = 7 #4 deng fen
generalization = "relu"

epoch_num = 300
im_size = 28
piece_length = int(im_size ** 2 / T)

#mimic pytorch F.linear
@torch._jit_internal.weak_script
def comp(input, state):
    Ax = torch.einsum('ij,bi->bj', matrix_A, input)

    if gen == "prod":
        Axh = torch.mul(Ax, state)
    if gen == "relu":
        Axh = torch.maximum(Ax, state)
        Axh = tf.maximum(Axh, 0)

    output = torch.einsum('ijk,bij->bk', tensor_G, Axh)
    return output


#mimic nn.linear
@weak_module
class grnn(Module):
    __constant__ = ['bias']

    def __int__(self, in_features, num_neurons_A, num_hidden, gen):
        super(grnn, self).__init__()
        self.num_neurons_A = num_neurons_A
        self.num_hidden = num_hidden
        self.in_features = in_features
        #self.out_features = out_features
        #num_hidden = R, num_neurons_A = M
        self.tensor_G = Parameter(torch.Tensor(num_neurons_A, num_hidden, num_hidden))
        self.matrix_A = Parameter(torch.Tensor(in_features, num_neurons_A))
        self.reset_parameter()
        self.gen = gen

    def reset_parameter(self):
        init.kaiming_uniform_(self.tensor_G, a=math.sqrt(5))
        init.kaiming_uniform_(self.matrix_A, a=math.sqrt(5))

    @weak_script_method
    #G_RNN 2019 ICLR paper equation (10)
    def forward(self, input, state):
        return comp(input, state)

    def extra_repr(self):
        return 'in_feature={}, out_feature={}, num_neurons_A={}, num_hidden={}'.format(self.in_features, self.matrix_A, self.tensor_G, self.out_features is not None)


class RNN(nn.Module):
#output_size only relative to the last FC layer, in mid gRNN layer, no output_size need to be mannuly specified
    def __int__(self, input_size, output_size, num_neurons_A, num_hidden, gen):
        super(RNN, self).__init__()
        #rnn_i is an object of class grnn
        self.rnn1 = grnn(input_size, num_neurons_A, num_hidden, gen)
        self.rnn2 = grnn(input_size, num_neurons_A, num_hidden, gen)
        self.rnn3 = grnn(input_size, num_neurons_A, num_hidden, gen)
        self.rnn4 = grnn(input_size, num_neurons_A, num_hidden, gen)
        self.rnn5 = grnn(input_size, num_neurons_A, num_hidden, gen)
        self.rnn6 = grnn(input_size, num_neurons_A, num_hidden, gen)
        self.rnn7 = grnn(input_size, num_neurons_A, num_hidden, gen)
        self.rnn8 = grnn(input_size, num_neurons_A, num_hidden, gen)
        self.rnn9 = grnn(input_size, num_neurons_A, num_hidden, gen)
        self.rnn10 = grnn(input_size, num_neurons_A, num_hidden, gen)
        self.rnn11 = grnn(input_size, num_neurons_A, num_hidden, gen)
        self.rnn12 = grnn(input_size, num_neurons_A, num_hidden, gen)
        self.rnn13 = grnn(input_size, num_neurons_A, num_hidden, gen)
        self.rnn14 = grnn(input_size, num_neurons_A, num_hidden, gen)
        self.rnn15 = grnn(input_size, num_neurons_A, num_hidden, gen)
        #self.rnn16 = grnn(input_size, num_neurons_A, num_hidden, gen)
        self.last = nn.Linear(input_size, num_output)

    def forward(self, input, hidden):
        hidden1 = self.rnn1(input, hidden)
        hidden2 = self.rnn2(input, hidden1)
        hidden3 = self.rnn3(input, hidden2)
        hidden4 = self.rnn4(input, hidden3)
        hidden5 = self.rnn5(input, hidden4)
        hidden6 = self.rnn6(input, hidden5)
        hidden7 = self.rnn7(input, hidden6)
        hidden8 = self.rnn8(input, hidden7)
        hidden9 = self.rnn9(input, hidden8)
        hidden10 = self.rnn10(input, hidden9)
        hidden11 = self.rnn11(input, hidden10)
        hidden12 = self.rnn12(input, hidden11)
        hidden13 = self.rnn13(input, hidden12)
        hidden14 = self.rnn14(input, hidden13)
        output = self.rnn15(input, hidden14)
        return output

    def initHidden1(self):
        return torch.zeros(1, self.num_hidden)

    def inithidden2(self):
        return torch.ones(1, self.num_hidden)

def onehott(label):
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


# some preprocessing function
# input shape [bsize,3,32,32], 3 represents RGB channel, take R
def rgb(images):
    image_np = images.numpy()
    image_input = np.zeros(((bsize, 32, 32)))
    for i in range(32):
        for j in range(32):
            # image_input[:,i,j] =(image_np[:,0,i,j] + image_np[:,1,i,j] + image_np[:,2,i,j])/3
            image_input[:, i, j] = image_np[:, 0, i, j]
    return image_input

def test(weights_TT):
    # Test part
    correct = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            images, labels = data
            images.to(device)
            labels.to(device)

            outputs = g_inner(weights_CP, images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.to(device) == labels.to(device)).sum().item()
    return 100.0 * (correct) / total

#instantiate the class
#input_size, output_size, num_neurons_A, num_hidden, gen
tt_rnn = RNN(num_neurons_A, 1, num_neurons_A, num_hidden, generalization)
learning_rate = 1e-4
print(learning_rate)

param_list = []
param_list.append(weights_CP)
optimizer = torch.optim.Adam(param_list, lr=learning_rate)

loss_list = []
wnorm_list = []
epoch_list = []
test_list = []

for epoch in range(epoch_num+1):
    loss_epoch = 0.0
    wnorm_epoch = 0.0
    for i, data in enumerate(trainloader, 0):

        inputs, labels = data
        inputs.to(device)

        if inputs.size()[0] != bsize:
            continue
        y = labels.float().to(device)
        y = onehott(y)

        if generalization == "relu":
            hidden = tt_rnn.initHidden1()
        else:
            hidden = tt_rnn.inithidden2()

        #omit tt_rnn.forward() args:input hidden
        y_predict = tt_rnn(inputs, hidden)
        #print('y_predict', y_predict[1,:], y_predict.size())

        m = torch.nn.Softmax()
        y_predict = m(y_predict)

        #print('y_predict_soft', y_predict[1,:], y_predict.size())
        #print('y', y[1,:], y.size())


        loss = (y_predict - y).pow(2).sum()
        loss_epoch += loss.item()
        print(epoch, i, "loss:", loss)
        print(' ')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("grad", torch.norm(weights_CP.grad.data).item())
        print("weights", torch.norm(weights_CP.data).item())
        wnorm_epoch += torch.norm(weights_CP.data).item()
        # with torch.no_grad():
        #     weights_CP -= learning_rate * weights_CP.grad
        #     weights_CP.grad.zero_()
    test_list.append(test(weights_CP))
    loss_list.append(loss_epoch)
    wnorm_list.append(wnorm_epoch)
    epoch_list.append(epoch)
    df = pd.DataFrame(data={'epoch': epoch_list, 'loss': loss_list, 'wnorm': wnorm_list, 'test':test_list})
    df.to_csv('mnist_cpkernel.csv')


print('Finished Training')