import torch
import torch.nn as nn
import torch.nn.functional as F
# this is for all the non-parametrized functions, like activations
# these are also included in the nn package, better to use the functional ones

class FCN(nn.Module):
    def __init__(self, IN_DIM, NUM_CLASSES):
        super().__init__()
        self.input_dim = IN_DIM
        self.num_classes = NUM_CLASSES
        self.fcn1 = nn.Linear(self.input_dim, 300)
        self.fcn2 = nn.Linear(300, self.num_classes)

    # now we need a forward passer method, on the input data
    def forward(self, x):
        # x would be the input data
        fcn1 = self.fcn1(x)
        fcn1 = F.relu(fcn1)
        fcn2 = self.fcn2(fcn1)
        output = torch.sigmoid(fcn2)  # y_pred
        return output


# if __name__ == '__main__':
#     # try to get in a quick check on random data
#     x = torch.randn(4, 784) # 4 training samples, of 784 input nodes, each
#     model = FCN(784, 10)
#     y_pred = model.forward(x)
#     print(y_pred.shape)
