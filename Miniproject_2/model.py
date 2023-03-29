###########                      Library import                    #############
from torch import empty, cat, arange, cuda
from torch.nn.functional import fold, unfold
import pickle
from pathlib import Path

from .others.layer import Conv2d, ReLu, NearestUpsampling, Sigmoid
from .others.MSE import MSE
from .others.sequential import Sequential

device = 'cuda' if cuda.is_available() else 'cpu'


###########                Net class : miniprojet 1                #############
class Model () :
    def __init__ (self) :
        ## instantiate model + optimizer + loss function + any other stuff you need
        self.learning_rate = 2
        self.batch_size = 50

        #super().__init__()

        self.MSE = MSE()
        self.sequential = Sequential(Conv2d(in_channels = 3, out_channels = 64, kernel_size = (3,3), stride=2, padding=1),
                                     ReLu(),
                                     Conv2d(in_channels = 64, out_channels = 128, kernel_size = (5,5), stride=2, padding=2),
                                     ReLu(),
                                     NearestUpsampling(in_channels = 128, out_channels = 64, kernel_size = (3,3), dilation = 0, padding=1, scale_factor = 2),
                                     ReLu(),
                                     NearestUpsampling(in_channels = 64, out_channels = 3, kernel_size = (3,3), dilation = 0, padding=1, scale_factor = 2),
                                     Sigmoid())

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        return self.sequential.forward(x)

    def backward(self, gradwrtoutput):
        self.sequential.backward(gradwrtoutput)


    def load_pretrained_model (self) :
        ## This loads the parameters saved in bestmodel .pth into the model
        model_path = Path(__file__).parent / "bestmodel.pth"
        bestmodel = open(model_path, 'rb')
        listParam = pickle.load(bestmodel)
        self.sequential.load(listParam)
        bestmodel.close()
        pass

    def save_trained_Model(self) :
        listParam = self.sequential.param()
        bestmodel = open('bestmodel.pth', 'wb')
        pickle.dump(listParam, bestmodel)
        bestmodel.close()
        pass

    def train (self, train_input, train_target, num_epochs) :
        #: train˙input : tensor of size (N, C, H, W) containing a noisy version of the images
        #: train˙target : tensor of size (N, C, H, W) containing another noisy version of the same images, which only differs from the input by their noise
        train_input, train_target = (train_input.float()/255.).to(device), (train_target.float()/255.).to(device)
        for i in range(num_epochs):
            print('Model 2 training : epoch',i+1)
            for batch in range(0,train_input.shape[0],self.batch_size):
                batch_input = train_input[batch:(batch+self.batch_size)]
                batch_target = train_target[batch:(batch+self.batch_size)]
                y = self.sequential(batch_input)
                Loss = MSE.forward(y,batch_target)
                dLoss = MSE.backward(y,batch_target)
                self.sequential.backward(dLoss)
                self.sequential.train(self.learning_rate)
        pass

    def predict (self, test_input) :
        #: test˙input : tensor of size (N1 , C, H, W) that has to be denoised by the trained or the loaded network .
        #: returns a tensor of the size (N1 , C, H, W)
        test_input = (test_input.float()/255.).to(device)
        return (((self.sequential.forward(test_input))*255).int()).cpu()

    def computePSNR(self, denoised, groundtruth) : # Peak Signal to Noise Ratio : denoised and ground˙truth have range [0 , 1]
        denoised, groundtruth = (denoised.float()/255).to(device), (groundtruth.float()/255).to(device)
        mse = ((denoised - groundtruth)**2).mean()
        return -10*((mse + 10**-8).log10()).cpu().detach()
