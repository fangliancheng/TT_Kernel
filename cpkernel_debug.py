import torch
import numpy as np
import pandas as pd
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.autograd import Variable

torch.backends.cudnn.deterministic = True
torch.manual_seed(9)

dtype = torch.float
device = torch.device("cpu")
device = torch.device("cuda:0") # Uncomment this to run on GPU

# Load data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# batch size
bsize = 250

T = 16
R = 64
M = 32
num_label = 10
epoch_num = 300
A = np.random.normal(0, 1, (M, 64))

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=bsize, shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=bsize, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def onehott(label):
    batch_size = bsize
    nb_digits = 10
    # Dummy input that HAS to be 2D for the scatter (you can use view(-1,1) if needed)
    #y = torch.LongTensor([[label[0]], [label[1]], [label[2]], [label[3]], [label[4]], [label[5]], [label[6]], [label[7]], [label[8]], [label[9]]])
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


def feature_map(X):
    num_split = np.sqrt(T)
    piece_length = int(X.shape[1] ** 2 / T)
    temp = np.zeros(((bsize, piece_length, T)))
    upper1 = np.split(np.split(X, num_split, axis=1)[0], num_split, axis=2)
    upper2 = np.split(np.split(X, num_split, axis=1)[1], num_split, axis=2)
    upper3 = np.split(np.split(X, num_split, axis=1)[2], num_split, axis=2)
    upper4 = np.split(np.split(X, num_split, axis=1)[3], num_split, axis=2)

    for i in range(0, bsize):
        for j in range(0, 4):
            temp[i, :, j] = (upper1[j])[i, :, :].flatten()
        for j in range(4, 8):
            temp[i, :, j] = (upper2[j-4])[i, :, :].flatten()
        for j in range(8, 12):
            temp[i, :, j] = (upper3[j-8])[i, :, :].flatten()
        for j in range(12, 16):
            temp[i, :, j] = (upper4[j-12])[i, :, :].flatten()

    #A = np.random.normal(0, 1, (M, piece_length))
    #rand a vetor: default is a row vector
    #b = np.random.rand(M).reshape(1, M)
    #f = np.ones(((bsize, M, T)))
    #f = np.random.normal(0, 1, (bsize, M, T))
    f = np.ones(((bsize, M, T)))
    for k in range(0, T):
        for j in range(0, bsize):
            # + care for broadcast!
            fm = np.matmul(A, temp[j, :, k]).reshape(1, M) #+ b
            #f[j, :, k] = torch.clamp(torch.from_numpy(fm), min=0)
            f[j,:,k] = torch.from_numpy(fm)
    #print("f shape:", f.shape)
    #print("f:", f)
    #print(f[bsize-1,:,T-1])
    return f


def g_inner(weights_CP, images):
    f = torch.from_numpy(feature_map(rgb(images))).float().to(device)
    y_pred = torch.zeros(num_label, bsize, R)
    # for batch_axis in range(0,bsize-1):
    for l in range(0, num_label):
        for r in range(0, R):
            temp = torch.zeros(bsize).to(device)
            for t in range(0, T):
                #print('f', f[:,:, t].size())
                #print('weisize', weights_CP[l, t, :, r].size())
                temp = torch.max(temp, torch.matmul(f[:, :, t], weights_CP[l, t, :, r]))
                #temp *= torch.matmul(f[:, :, t], weights_CP[l, t, :, r])
                #print(temp)
                #print('aftemp', temp.size())
            y_pred[l, :, r] = temp

    #y_predict = torch.matmul(y_pred, torch.ones(R, 1))
    y_predict = torch.sum(y_pred, 2)
    #y_predict is a vector
    return torch.t(y_predict).to(device)


def test(weights_CP):
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

# randomly generate the component of CP decomposition
#weights_CP = torch.randn(num_label, T, M, R, device=device, dtype=dtype, requires_grad=True)
weights_CP = Variable(torch.randn(num_label, T, M, R, device=device, dtype=dtype), requires_grad=True)


learning_rate = 0.0001
print(learning_rate)

param_list = []
param_list.append(weights_CP)
optimizer = torch.optim.SGD(param_list, lr=learning_rate)#, momentum=0.9)

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

        y_predict = g_inner(weights_CP, inputs)
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
    df.to_csv('cpkernel.csv')


print('Finished Training')




