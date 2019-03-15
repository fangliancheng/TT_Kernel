import torch

#mimic pytorch F.linear
##########################
#@torch._jit_internal.weak_script
def comp(input, state):
    Ax = torch.einsum('ij,bi->bj', matrix_A, input)

    if gen == "prod":
        Axh = torch.mul(Ax, state)
    if gen == "relu":
        Axh = torch.maximum(Ax, state)
        Axh = torch.maximum(Axh, 0)

    output = torch.einsum('ijk,bij->bk', tensor_G, Axh)
    return output, output
