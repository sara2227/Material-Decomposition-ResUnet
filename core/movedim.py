import torch

class MoveDimTransform(object):
    def __init__(self, dim1, dim2):
        self.dim1 = dim1
        self.dim2 = dim2

    def __call__(self, tensor):
        return torch.movedim(tensor, self.dim1, self.dim2)