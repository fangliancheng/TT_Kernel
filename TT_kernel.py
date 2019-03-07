import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import tensorly as tl
from math import ceil
from torch.autograd import Variable
#CP decomposition
from tensorly.decomposition import parafac

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                          shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# get a random image
dataiter = iter(trainloader)
images, labels = dataiter.next()
#print(images.size())

#shape: 1 3 32 32
#take average over RGB channels
def aver(image):
    image_np = image.numpy()
    image_input = np.zeros((32,32))
    for i in range(0,31):
         for j in range(0,31):
             image_input[i,j] =(image_np[0,0,i,j] + image_np[0,1,i,j] + image_np[0,2,i,j])/3
    return image_input


"""
#input x should be a numpy array
def relu(x):
    for i in x.size():
        if x[i] < 0:
            x[i] = 0
    return x
 """

#Test average
#print(aver(images).shape)

#Input: numpy 2-D matrix(array), e.g Cifer-10, 32*32 matrix (RGB?)
#First,devide it into window-patches,then reshape to column vectors, pick window size 16*16, no overlap, finally we get (x_1,...,x_4). x_i \in R^256
#Second, pick a local feature map for each x_i, f:R^256--->R^32, the global feature map is defined as a tensor \in R^{256*256...*256} #64, see E.M paper (2)
#Here we pick local feature map as ReLU(Ax+b), A,b are hyper-parameters. Return global feature map(a tensor)
#We pick a rather huge window and map it to lower dimensional feature space, but this still leads to a quite big feature tensor
def feature_map(X):

    #stride = 16.0
    #num_split = X.shape[0]/stride
    temp = np.zeros((256,4))
    #print(temp.shape)
    """
    upper1 = np.hsplit(np.vsplit(mat, num_split)[0], num_split)
    upper2 = np.hsplit(np.vsplit(mat, num_split)[1], num_split)
    upper3 = np.hsplit(np.vsplit(mat, num_split)[2], num_split)
    upper4 = np.hsplit(np.vsplit(mat, num_split)[3], num_split)
    upper5 = np.hsplit(np.vsplit(mat, num_split)[4], num_split)
    upper6 = np.hsplit(np.vsplit(mat, num_split)[5], num_split)
    upper7 = np.hsplit(np.vsplit(mat, num_split)[6], num_split)
    upper8 = np.hsplit(np.vsplit(mat, num_split)[7], num_split)
    for i in range(0,7):
        temp[:,i] = upper1[i].flatten()
    for i in range(8,15):
        temp[:,i] = upper2[i - 8].flatten()
    for i in range(16,23):
        temp[:,i] = upper3[i - 16].flatten()
    for i in range(24,31):
        temp[:,i] = upper4[i - 24].flatten()
    for i in range(32,39):
        temp[:,i] = upper5[i - 32].flatten()
    for i in range(40,47):
        temp[:,i] = upper6[i - 40].flatten()
    for i in range(48,55):
        temp[:,i] = upper7[i - 48].flatten()
    for i in range(56,63):
        temp[:,i] = upper8[i - 56].flatten()
    """

    upper_half = np.hsplit(np.vsplit(X, 2)[0], 2)
    lower_half = np.hsplit(np.vsplit(X, 2)[1], 2)

    upper_left = upper_half[0]
    upper_right = upper_half[1]
    lower_left = lower_half[0]
    lower_right = lower_half[1]

    temp[:,0] = upper_left.flatten()
    temp[:,1] = upper_right.flatten()
    temp[:,2] = lower_left.flatten()
    temp[:,3] = lower_right.flatten()
    #print(temp[:,0].shape)
    #print(temp[:,0].size)
    A = np.random.rand(32,256)
    b = np.random.rand(32,1)

    #print("b shape",b.shape)
    #print("b' shape", b.T.shape)
    f = np.zeros((32,4))
    for k in range(0,3):
    #f[:,k] = relu(A*temp[:,k]+b)

        fm = np.matmul(A , temp[:,k]) + b.T

        f[:,k] = torch.clamp(torch.from_numpy(fm), min = 0)
    """
    feature_tensor = np.zeros(32,32,32,32)
    for i in range(0,31):
        for j in range(0,31):
            for l in range(0,31):
                for m in range(0,31):
                    feature_tensor[i,j,k,m] = f[i,0]*f[j,1]*f[l,2]*f[m,3]

    return feature_tensor
    """
    #f(x^t) in each column
    return f

#randomly initialize learnable weights W
#random number from a normal distribution
#input to tl.tensor must be a numpy array
temp = torch.randn(32,32,32,32)
W_full = tl.tensor(temp.numpy())

#Applying CP decomposition to W
from tensorly.decomposition import parafac
factor = parafac(W_full,rank =3)
#factor would be a list consists 4 matrixs, each one is 32*3 ,each column represent one term in CP-decomposition
#factor is a list

#w_i is a 32*4 matrix, each column is one component of CP decomposition
#w1 is convert to torch tensor form, ready to use Variable to set 'requires_grad'
w1 = torch.from_numpy(factor[0])
w2 = torch.from_numpy(factor[1])
w3 = torch.from_numpy(factor[2])
w4 = torch.from_numpy(factor[3])
w1 = Variable(w1, requires_grad = True)
w2 = Variable(w2, requires_grad = True)
w3 = Variable(w3, requires_grad = True)
w4 = Variable(w4, requires_grad = True)

learning_rate = 1e-6

str = input("which mode:")
print(str)

if str == '1':
    for epoch in range(3):

        for i,data in enumerate(trainloader,0):
            inputs,labels = data

            # f is a matrix, each column is f(x^t), t = 1,2,3,4
            #torch.dot 's argument must float tensor, but when transform from numpy to torch, default is double tensor
            f = torch.from_numpy(feature_map(aver(inputs))).float()
            y = labels.float()


            #normal multiplication between inner product: coresponds to sum-product NN
            y_pred1 = torch.dot(w1[:,0],f[:,0]) * torch.dot(w2[:,0],f[:,1]) * torch.dot(w3[:,0],f[:,2]) * torch.dot(w4[:,0],f[:,3])
            y_pred2 = torch.dot(w1[:,1],f[:,0]) * torch.dot(w2[:,1],f[:,1]) * torch.dot(w3[:,1],f[:,2]) * torch.dot(w4[:,1],f[:,3])
            y_pred3 = torch.dot(w1[:,2],f[:,0]) * torch.dot(w2[:,2],f[:,1]) * torch.dot(w3[:,2],f[:,2]) * torch.dot(w4[:,2],f[:,3])
            y_pred  = y_pred1 + y_pred2 + y_pred3

            loss = (y_pred - y).pow(2)
            print(epoch, i ,"loss:", loss.item())

            loss.backward()

            with torch.no_grad():
                w1 -= learning_rate * w1.grad
                w2 -= learning_rate * w2.grad
                w3 -= learning_rate * w3.grad
                w4 -= learning_rate * w4.grad

                w1.grad.zero_()
                w2.grad.zero_()
                w3.grad.zero_()
                w4.grad.zero_()

if str == '2':
    for epoch in range(3):

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            # f is a matrix, each column is f(x^t), t = 1,2,3,4
            # torch.dot 's argument must float tensor, but when transform from numpy to torch, default is double tensor
            f = torch.from_numpy(feature_map(aver(inputs))).float()
            y = labels.float()

            # normal multiplication between inner product: coresponds to sum-product NN
            y_pred1 = torch.max(torch.max(torch.max(torch.dot(w1[:,0],f[:,0]).clamp(min = 0),torch.dot(w2[:,0],f[:,1])),torch.dot(w3[:,0],f[:,2]) ),torch.dot(w4[:,0],f[:,3]))
            y_pred2 = torch.max(torch.max(torch.max(torch.dot(w1[:,1],f[:,0]).clamp(min = 0),torch.dot(w2[:,1],f[:,1])),torch.dot(w3[:,1],f[:,2]) ),torch.dot(w4[:,1],f[:,3]))
            y_pred3 = torch.max(torch.max(torch.max(torch.dot(w1[:,2],f[:,0]).clamp(min = 0),torch.dot(w2[:,2],f[:,1])),torch.dot(w3[:,2],f[:,2]) ),torch.dot(w4[:,2],f[:,3]))
            y_pred = y_pred1 + y_pred2 + y_pred3

            loss = (y_pred - y).pow(2).sum()
            print(epoch, i, "loss:", loss.item())

            loss.backward()

            with torch.no_grad():
                w1 -= learning_rate * w1.grad
                w2 -= learning_rate * w2.grad
                w3 -= learning_rate * w3.grad
                w4 -= learning_rate * w4.grad

                w1.grad.zero_()
                w2.grad.zero_()
                w3.grad.zero_()
                w4.grad.zero_()














