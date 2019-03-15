import torch
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch.nn import init
from vF import comp

#mimic nn.linear
#@weak_module
class Grnn(Module):
    #__constant__ = ['bias']

    def __int__(self, in_features, num_neurons_A, num_hidden, gen):
        super(Grnn, self).__init__()
        self.num_neurons_A = num_neurons_A
        self.num_hidden = num_hidden
        self.in_features = in_features
        self.tensor_G = Parameter(torch.Tensor(num_neurons_A, num_hidden, num_hidden))
        self.matrix_A = Parameter(torch.Tensor(in_features, num_neurons_A))
        self.reset_parameter()
        self.gen = gen

    def reset_parameter(self):
        init.kaiming_uniform_(self.tensor_G, a=math.sqrt(5))
        init.kaiming_uniform_(self.matrix_A, a=math.sqrt(5))

    #@weak_script_method
    #G_RNN 2019 ICLR paper equation (10)
    def forward(self, input, state):
        return comp(input, state)

    def extra_repr(self):
        return 'in_feature={}, out_feature={}, num_neurons_A={}, num_hidden={}'.format(self.in_features, self.matrix_A, self.tensor_G, self.out_features is not None)
