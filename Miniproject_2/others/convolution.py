from torch import empty, cat, arange
from torch.nn.functional import fold, unfold

class Conv2d():
        """
        2D convolution Module
        ------------------------------------------------------------------------

        Variables
        ------------------------------------------------------------------------
        in_channels : (int) Number of channels in the input image

        out_channels : (int) Number of channels produced by the convolution

        kernel_size : (int) or (tuple) Size of the convolving kernel

        stride : (int) or (tuple) Stride of the convolution. Default: 1

        padding :  (int) or (tuple) Padding added to all four sides of the input. Default: 0

        weight : (float) tensor of shape (output_dim, input_dim)
            Contains weights of the layer

        bias : (float) tensor of shape (output_dim)
            Contains biases of the layer

        dWeight : (float) tensor of shape (output_dim, input_dim)
            Contains derivative of the loss wrt the weights of the layer

        dBias : (float) tensor of shape (output_dim)
            Contains derivative of the loss wrt the biases of the layer


        Functions
        ------------------------------------------------------------------------
        forward(x)
            return the convolution of x

        backward(dL_dS)
            computes and saves the derivatives of the loss wrt weights, biases,
            and the layer inputs

        update_parameters(eta)
            updates parameters along the opposite gradient directions

        """
    def __init__ (self, in_channels, out_channels, kernel_size, stride=1, padding=0):

        #input storage
        self.al

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if(type(padding) is int):
            self.padding = [padding, padding]
        elif (type(padding) is tuple):
            self.padding = [padding[0], padding[1]]
        else :
            raise TypeError('Wrong type for padding, takes int or tupple only')
        if(type(stride) is int):
            self.stride = [stride, stride]
        elif (type(stride) is tuple):
            self.stride = [stride[0], stride[1]]
        else :
            raise TypeError('Wrong type for stride, takes int or tupple only')

        self.weight = empty(out_channels, in_channels, kernel_size[0], kernel_size[1])
        self.bias = empty(out_channels)
        self.dWeight = empty(out_channels, in_channels, kernel_size[0], kernel_size[1])
        self.dBias = empty(out_channels)

    def __call__(self, *al):
        self.forward(al)

    def forward(self, *al):
        unfolded = unfold(al, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
        self.weight.view(out_channels, -1) @ unfolded + self.bias.view(1, -1, 1)
        wxb = self.weight.view(out_channels, -1) @ unfolded + self.bias.view (1, -1, 1)
        return wxb.view(x.shape[0], out_channels, -(-(x.shape[2] + 2*self.padding[0] - self.kernel_size[0] - 2)//self.stride[0]) + 1, -(-(x.shape[3] + 2*padding[1] - kernel_size[1] - 2)//stride[1]) + 1)

    def backward(self, *dL_dzl):
        #
        self.dBias = (dL_dzl.sum((2,3))).mean(0)

        #dL_dw = dL_dzl * dzl_dw = convolution(al, dL_dzl)
        self.dL_dw =
        return dx


        raise NotImplementedError

    def sgd(eta):
        self.weight = self.weight - self.dWeight*eta
        self.bias = self.bias - self.dBias*eta

    def param(self):
        return [self.weight, self.bias, self.dWeight, self.dBias]
