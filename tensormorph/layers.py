# -*- coding: utf-8 -*-

from tpr import *


# xxx broken at sigmoid boundaries
class ResidualSoftmax(nn.Module):

    def __init__(self, input1_size, input2_size=None, output_size):
        super(ResidualSoftmax, self).__init__()
        # Residual connection from input1, zero out (optional) input2
        self.input1_size = input1_size
        if input2_size is not None:
            self.input2_size = input2_size
        else:
            self.input2_size = 0
        # Unrestricted linear map to model departures from identity
        self.linear = nn.Linear(input1_size + input2_size, output_size)
        #self.linear.data.fill_(-3.0)
        #self.linear.bias.fill_(-3.0)

    def forward(self, input1, input2=None):
        if input2 is None:
            X = input1
        else:
            X = torch.cat([input1, 0.0 * input2], dim=-1)
        # Linear map + approximate identity under sigmoid
        Y = self.linear(X) + 3.0 * X
        Y = sigmoid(Y)
        return Y
