from torch import empty, cat, arange
from torch.nn.functional import fold, unfold
#from other import layer

class Sequential(object):
    def __init__ (self, *layers):
        self.layers = []
        for layer in layers:
            self.layers.append(layer)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def backward(self, gradwrtoutput):
        for layer in reversed(self.layers):
            gradwrtoutput = layer.backward(gradwrtoutput)

    def train(self, eta):
        for layer in self.layers:
            x = layer.train(eta)

    def param(self):
        listParam = list()
        for layer in self.layers :
            listParam = listParam + (layer.param())
        return listParam

    def load(self, listParam):
        for layer in reversed(self.layers):
            listParam = layer.load(listParam)
