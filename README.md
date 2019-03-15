# TT_Kernel

Reference: 

1.Supervised Learning with Tensor Networks E.M.Stoudenmire 

2.Generalized Tensor Models for Recurrent Neural Networks 

3.Convolutional Rectifier Networks as Generalized Tensor Decompositions Nadav Cohen

Main Idea: 
In supervised learning, we consider a naive linear classifer:

\mathcal{l}(x) = <W,\phi(x)>, where \phi(x) is the outerproduct of many patches of x

In [1], they proposed to TT decompose the W, which can reduce parameter number, update certain part of W in each training iteration.

In [2], they proved that if we futher replace scalar product of the score function by any generalized operation(e.g max{x,y,0}), the score function coresponds to the output of a particular RNN.
