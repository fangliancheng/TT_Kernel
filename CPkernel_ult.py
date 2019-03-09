import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.autograd import Variable

dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU

# Load data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# batch size
bsize = 250
learning_rate = 1e-4
T = 16
R = 64
M = 32
epoch_num = 3

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=bsize, shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=bsize, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# some preprocessing function
# input shape [bsize,3,32,32], 3 represents RGB channel, take R
def rgb(images):
    image_np = images.numpy()
    image_input = np.zeros(((bsize, 32, 32)))
    for i in range(0, 31):
        for j in range(0, 31):
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

    for i in range(0, bsize - 1):
        for j in range(0, 3):
            temp[i, :, j] = (upper1[j])[i, :, :].flatten()
        for j in range(4, 7):
            temp[i, :, j] = (upper2[j-4])[i, :, :].flatten()
        for j in range(8, 11):
            temp[i, :, j] = (upper3[j-8])[i, :, :].flatten()
        for j in range(12, 15):
            temp[i, :, j] = (upper4[j-12])[i, :, :].flatten()

    A = np.random.rand(M, piece_length)
    b = np.random.randn(M, 1)

    f = np.zeros(((bsize, M, T)))
    for k in range(0, T - 1):
        for j in range(0, bsize - 1):
            fm = np.matmul(A, temp[j, :, k]) + b.T
            f[j, :, k] = torch.clamp(torch.from_numpy(fm), min=0)
    return f


# Ouput f is bsize * M * T

# forward function
# argument:W:CP decomposition elements f:feature map
def inner(weights_CP, images):
    f = torch.from_numpy(feature_map(rgb(images))).float()
    y_pred = torch.randn(bsize, R)
    # for batch_axis in range(0,bsize-1):
    # replace by lift a dimension and do matrix multiplication
    for rank in range(0, R - 1):
        temp = 1
        for t in range(0, T - 1):
            temp = temp * torch.matmul(f[:, :, t], weights_CP[t, :, rank])
        y_pred[:, rank] = temp

    y_predict = torch.matmul(y_pred, torch.ones(R, 1))
    return y_predict


def g_inner(weights_CP, images):
    f = torch.from_numpy(feature_map(rgb(images))).float()
    y_pred = torch.randn(bsize, R)
    # for batch_axis in range(0,bsize-1):
    for r in range(0, R - 1):
        temp = torch.zeros(1)
        for t in range(1, T - 1):
            temp = torch.max(temp, torch.matmul(f[:, :, t], weights_CP[t, :, r]))
        y_pred[:, r] = temp

    y_predict = torch.matmul(y_pred, torch.ones(R, 1))
    return y_predict


# randomly generate the component of CP decomposition
weights_CP = torch.randn(T, M, R)
for t in range(0, T - 1):
    #weights_CP[t, :, :] = Variable(torch.randn(M, R), requires_grad=True)
    weights_CP[t, :, :] = torch.rand(M, R, requires_grad=True)

str = input("which mode? mode1: sum-product NN, mode2: shollow CNN")
print(str)

if str == '1':
    sss = input("which training method?")
    print(sss)
    if sss == "1":
        for epoch in range(epoch_num):
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                if (inputs.size()[0] != bsize):
                    continue
                y = labels.float()
                y_predict = inner(weights_CP, inputs)
                loss = (y_predict - y).pow(2).sum()
                print(epoch, i, "loss:", loss.item())

                loss.backward()

                with torch.no_grad():
                    for i in range(0, T - 1):
                        weights_CP[i, :, :] -= learning_rate * weights_CP[i, :, :].grad
                    for i in range(0, T - 1):
                        weights_CP[i, :, :].grad.zero_()

        print('Finished Training')

    else:
        param_list = []
        param_list.append(weights_CP)
        optimizer = torch.optim.Adam(param_list, lr=learning_rate)
        for epoch in range(epoch_num):

            for i, data in enumerate(trainloader, 0):
                inputs, labels = data

                if (inputs.size()[0] != 32):
                    continue
                y = labels.float()
                y_predict = inner(weights_CP, inputs)
                loss = (y_predict - y).pow(2).sum()
                print(epoch, i, "loss:", loss.item())
                optimizer.zero_grid()

                loss.backward()

                optimizer.step()

    # Test part
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = inner(weights_CP, images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

if str == '2':
    sss = input("which training method?")
    print(sss)
    if sss == "1":
        for epoch in range(epoch_num):
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data

                if (inputs.size()[0] != bsize):
                    continue
                y = labels.float()
                y_predict = g_inner(weights_CP, inputs)
                loss = (y_predict - y).pow(2).sum()
                print(epoch, i, "loss:", loss.item())

                loss.backward()

                with torch.no_grad():
                    for p in range(0, T - 1):
                        weights_CP[i, :, :] -= learning_rate * weights_CP[p, :, :].grad
                    for q in range(0, T - 1):
                        weights_CP[q, :, :].grad.zero_()

        print('Finished Training')

    else:
        param_list = []
        param_list.append(weights_CP)
        optimizer = torch.optim.Adam(param_list, lr=learning_rate)
        for epoch in range(epoch_num):
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data

                if (inputs.size()[0] != 32):
                    continue
                y = labels.float()
                y_predict = g_inner(weights_CP, inputs)
                loss = (y_predict - y).pow(2).sum()
                print(epoch, i, "loss:", loss.item())
                optimizer.zero_grid()

                loss.backward()

                optimizer.step()

    # Test part
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = g_inner(weights_CP, images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))


