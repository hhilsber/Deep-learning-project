from torch import empty, cat, arange, cuda
from torch.nn.functional import fold, unfold

class Conv2d(object):
    """
    2D convolution Module
    ----    Variables   ---- :
    in_channels : (int) Number of channels in the input image
    out_channels : (int) Number of channels produced by the convolution
    kernel_size : (int) or (tuple) Size of the convolving kernel
    stride : (int) or (tuple) Stride of the convolution. Default: 1
    padding :  (int) or (tuple) Padding added to all four sides of the input. Default: 0
    weight : (float) tensor of shape (output_dim, input_dim)
        Contains weights of the layer
    bias : (float) tensor of shape (output_dim)
        Contains biases of the layer
    dL_dw : (float) tensor of shape (output_dim, input_dim)
        Contains derivative of the loss wrt the weights of the layer
    dL_db : (float) tensor of shape (output_dim)
        Contains derivative of the loss wrt the biases of the layer

    ----    Functions   ---- :
    forward(al)
        return zl+1 the convolution of al
    backward(gradwrtoutput)
        computes and saves the derivative of weight and bias and return the layer derivative dL_dal
    train(eta)
        updates weights and bias with SGD
    """
    def __init__ (self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):

        self.device = 'cuda' if cuda.is_available() else 'cpu'

        #input storage
        self.alp = None

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if(type(kernel_size) is int):
            self.kernel_size = [kernel_size, kernel_size]
        elif (type(kernel_size) is tuple):
            self.kernel_size = [kernel_size[0], kernel_size[1]]
        else :
            raise TypeError('Wrong type for padding, takes int or tupple only')

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

        k = 1/(in_channels*self.kernel_size[0]*self.kernel_size[1])

        self.weight = (empty(out_channels, in_channels, self.kernel_size[0], self.kernel_size[1]))
        self.bias = (empty(out_channels))
        self.dL_dw = (empty(out_channels, in_channels, self.kernel_size[0], self.kernel_size[1]))
        self.dL_db = (empty(out_channels))

        self.weight = (self.weight.uniform_(-(k**0.5), k**0.5)).to(self.device)
        self.bias = (self.bias.uniform_(-(k**0.5), k**0.5)).to(self.device)
        self.dL_dw = self.dL_dw.to(self.device)
        self.dL_db = self.dL_db.to(self.device)

    def __call__(self, alp):
        return self.forward(alp)

    def forward(self, alp):
        self.alp = alp
        unfolded = unfold(alp, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
        wxb = (self.weight.view(self.out_channels, -1)) @ unfolded + self.bias.view (1, -1, 1)
        zlp = wxb.view(alp.shape[0], self.out_channels, (alp.shape[2] + 2*self.padding[0] - self.kernel_size[0] )//self.stride[0] + 1, (alp.shape[3] + 2*self.padding[1] - self.kernel_size[1])//self.stride[1] + 1)
        return zlp

    def backward(self, dL_dzl):
        self.dL_db = dL_dzl.sum((2,3)).sum(0)

        unfolded = unfold(self.alp, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
        dL_dzl_lin = dL_dzl.reshape(dL_dzl.shape[0],dL_dzl.shape[1],-1)
        self.dL_dw = ((dL_dzl_lin @ unfolded.transpose(1,2)).sum(0)).reshape(self.weight.shape)

        xdL = ((self.weight.view(self.out_channels, -1)).permute(1,0)) @ (dL_dzl.view(dL_dzl.shape[0], self.out_channels,-1))
        dL_dalp = fold(xdL,(self.alp.shape[2],self.alp.shape[3]), kernel_size = self.kernel_size, dilation=1, padding=self.padding, stride=self.stride)
        return dL_dalp

    def train(self, eta):
        self.weight = self.weight - self.dL_dw*eta
        self.bias = self.bias - self.dL_db*eta

    def param(self):
        return [(self.weight.cpu(), self.dL_dw.cpu()), (self.bias.cpu(), self.dL_db.cpu())]

    def load(self, listParam):
        bias = listParam.pop()
        weight = listParam.pop()
        self.weight = (weight[0]).to(self.device)
        self.dL_dw = weight[1].to(self.device)
        self.bias = bias[0].to(self.device)
        self.dL_db = bias[1].to(self.device)
        return listParam


class ReLu(object):
    def __init__ (self):
        # a(l) = relu(z(l))
        self.zl = None

    def __call__(self, zl):
        return self.forward(zl)

    def forward(self,zl):
        #input = z(l)
        #output a(l)
        self.zl = zl
        return (zl*(zl > 0))
        #return zl[zl<0]

    def backward(self,dl_dal):
        #input = dLoss/d_al
        dzl = self.zl
        dzl[dzl<=0] = 0
        dzl[dzl>0] = 1

        dl_dzl = dl_dal.mul(dzl)
        return dl_dzl

    def train(self, eta):
        return

    def load(self, listParam):
        return listParam

    def param(self):
        return []


class Sigmoid(object):
    def __init__ (self):
        # a(l) = sigmoid(z(l))
        self.zl = 0

    def __call__(self, zl):
        return self.forward(zl)

    def forward(self,zl):
        #input = zl
        self.zl = zl
        al = self.zl.sigmoid()
        return al

    def backward(self,dl_dal):
        #input = dLoss/d_al
        dzl = self.zl
        dzl = dzl.sigmoid()*(1-dzl.sigmoid())
        dl_dzl = dl_dal.mul(dzl)
        return dl_dzl

    def train(self, eta):
        return

    def param(self):
        return []

    def load(self, listParam):
        return listParam


class UpSampleNN(object):
    def __init__ (self, scale_factor):
        self.alp = None
        self.scale_factor = scale_factor


    def __call__(self, al):
        return self.forward(al)

    def forward(self,alp):
        self.alp = alp
        zl = alp.repeat_interleave(self.scale_factor, dim=3)
        zl = zl.repeat_interleave(self.scale_factor, dim=2)
        return zl

    def backward(self, dL_dxpr):
        unfolded = unfold(dL_dxpr,kernel_size=self.scale_factor, stride = self.scale_factor)
        unfolded = unfolded.reshape([dL_dxpr.shape[0], dL_dxpr.shape[1], 2*self.scale_factor,-1])
        sum = unfolded.sum(2)
        dL_dx = sum.reshape((dL_dxpr.shape[0],dL_dxpr.shape[1], int(dL_dxpr.shape[2]/self.scale_factor), int(dL_dxpr.shape[3]/self.scale_factor)))
        return dL_dx

    def train(self, eta):
        return

    def param(self):
        return []

    def load(self, listParam):
        return listParam

class NearestUpsampling(object):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, padding, scale_factor):

        self.conv = Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride=1, padding=padding)
        self.upsample = UpSampleNN(scale_factor)

    def __call__(self, al):
        return self.forward(al)

    def forward(self,alp):
        zlpr = self.upsample(alp)
        return self.conv(zlpr)

    def backward(self, dL_dzl):
        dl_dxpr = self.conv.backward(dL_dzl)
        return self.upsample.backward(dl_dxpr)

    def train(self, eta):
        self.conv.train(eta)
        return

    def param(self):
        return self.conv.param()

    def load(self, listParam):
        return self.conv.load(listParam)
