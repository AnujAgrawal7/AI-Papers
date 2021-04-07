import torch.nn as nn
import torch.nn.functional as F

class AlexNetwork(nn.Module):
    def __init__(self):
        super(AlexNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels= 3, out_channels=96, kernel_size= (11,11), stride= (4,4), padding= (0,0))
        self.n1 = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.maxpool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.maxpool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.conv2 = nn.Conv2d(in_channels= 96, out_channels= 256, kernel_size= (5,5), stride= (1,1), padding = (2,2))
        self.n2 = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2)
        self.conv3 = nn.Conv2d(in_channels= 256, out_channels= 384, kernel_size= (3,3), stride= (1,1), padding = (1,1))
        self.conv4 = nn.Conv2d(in_channels= 384, out_channels= 384, kernel_size= (3,3), stride= (1,1), padding = (1,1))
        self.conv5 = nn.Conv2d(in_channels= 384, out_channels= 256, kernel_size= (3,3), stride= (1,1), padding = (1,1))
        self.fc1 = nn.Linear(6*6*256, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.dpt1 = nn.Dropout()
        self.dpt2 = nn.Dropout()
        self.opt = nn.Linear(4096, 10)
        self.sm = nn.Softmax(dim= 1)

    def forward(self, x):
        x = self.n1(F.relu(self.conv1(x)))
        x = self.maxpool1(x)
        x = self.n2(F.relu(self.conv2(x)))
        x = self.maxpool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.maxpool3(x)
        x = x.view(-1, 6*6*256)
        x = self.dpt1(x)
        x = self.dpt2(F.relu(self.fc1(x)))
        x = F.relu(self.fc2(x))
        x = self.sm(self.opt(x))
        return x




