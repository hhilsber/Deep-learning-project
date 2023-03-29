###########                Library import                #############
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path

device = 'cuda' if torch.cuda.is_available() else 'cpu'

###########           Net class : miniprojetc 1          #############
class Model (nn.Module) :
    def __init__ (self) :
        ## instantiate model + optimizer + loss function + any other stuff you need
        self.learning_rate = 0.001
        self.mini_batch_size = 200

        super().__init__()

        self.conv3_32 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv32_32_1 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv32_32_2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv32_64_x = nn.Conv2d(32, 64, 3, padding=1)
        self.conv64_96_x = nn.Conv2d(64, 96, 3, padding=1)
        self.conv64_96_y = nn.Conv2d(64, 96, 3, padding=1)
        self.conv96_96_1 = nn.Conv2d(96, 96, 3, padding=1)
        self.conv96_64_x = nn.Conv2d(96, 64, 3, padding=1)
        self.conv96_64_y = nn.Conv2d(96, 64, 3, padding=1)
        self.conv64_32 = nn.Conv2d(64, 32, 3, padding=1)
        self.t_conv32_3 = nn.ConvTranspose2d(32, 3, 1, stride=1)
        self.bn32_1 = nn.BatchNorm2d(32)
        self.bn32_2 = nn.BatchNorm2d(32)
        self.bn96 = nn.BatchNorm2d(96)
        self.bn96_y = nn.BatchNorm2d(96)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), self.learning_rate)

    def forward(self, x):
        if(x.dtype is torch.uint8):
          x= x.float()/255
        x = F.relu(self.conv3_32(x))
        x = F.relu(self.conv32_32_1(x))
        x = self.bn32_1(x)
        x = F.relu(self.conv32_64_x(x))
        y = F.relu(self.conv64_96_y(x))
        y = F.relu(self.conv96_96_1(y))
        y = self.bn96_y(y)
        y = F.relu(self.conv96_64_y(y))
        x = x+y
        x = F.relu(self.conv64_96_x(x))
        x = self.bn96(x)
        x = F.relu(self.conv96_64_x(x))
        x = F.relu(self.conv64_32(x))
        x = self.bn32_2(x)
        x = F.relu(self.conv32_32_2(x))
        x = self.t_conv32_3(x)
        x = F.relu(x)
        x = torch.clamp(x, min=0, max=1)
        return x
    def load_pretrained_model (self) :
        ## This loads the parameters saved in bestmodel .pth into the model
        #self.load_state_dict(torch.load(path, map_location=torch.device(device)))
        model_path = Path(__file__).parent / "bestmodel.pth"
        self.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        pass

    def save_trained_Model(self) :
        torch.save(self.state_dict(), 'deep_learning/Miniproject_1/bestmodel.pth')
        pass

    def train (self, train_input, train_target, num_epochs) :
        #: train˙input : tensor of size (N, C, H, W) containing a noisy version of the images
        #: train˙target : tensor of size (N, C, H, W) containing another noisy version of the same images, which only differs from the input by their noise
        train_input, train_target = (train_input.float()/255).to(device), (train_target.float()/255).to(device)
        for epoch in tqdm(range(num_epochs)):
            for batch in range(0, train_input.size(0), self.mini_batch_size):
                output = self(train_input.narrow(0, batch, self.mini_batch_size))
                loss = self.criterion(output, train_target.narrow(0, batch, self.mini_batch_size))
                self.zero_grad()
                loss.backward()
                self.optimizer.step()
        pass

    def predict (self, test_input) :
        #: test˙input : tensor of size (N1 , C, H, W) that has to be denoised by the trained or the loaded network .
        #: returns a tensor of the size (N1 , C, H, W)
        test_input = (test_input.float()/255.).to(device)
        return ((self(test_input)*255).int()).cpu().detach()

    def computePSNR(self, denoised, groundtruth) :
        # Peak Signal to Noise Ratio : denoised and ground˙truth have range [0 , 1]
        denoised, groundtruth = (denoised.float()/255).to(device), (groundtruth.float()/255).to(device)
        mse = ((denoised - groundtruth)**2).mean()
        return -10*((mse + 10**-8).log10()).cpu().detach()
