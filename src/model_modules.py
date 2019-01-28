import torch
import torch.nn as nn
import numpy as np

class DownBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        return self.relu(out)

class UpBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, stride=1, output_padding=1):
        super().__init__()

        self.convt1 = nn.ConvTranspose2d(in_c, out_c, kernel_size=kernel_size, stride=stride, output_padding=output_padding)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.convt1(x)
        out = self.bn1(out)
        return self.relu(out)

class Encoder(nn.Module):
    def __init__(self, in_c=3, z_dim=512):
        super().__init__()
        self.z_dim = z_dim
        self.in_c = in_c

        self.conv1 = DownBlock(in_c, 10, kernel_size=9, stride=1)
        self.conv2 = DownBlock(10, 20, kernel_size=7, stride=3)
        self.conv3 = DownBlock(20, 40, kernel_size=5, stride=1)
        self.conv4 = DownBlock(40, 80, kernel_size=3, stride=3) # (b, 80, 4, 4)
        self.dropout = nn.Dropout2d(0.5)
        self.fc5 = nn.Linear(1280, 1024)
        self.relu = nn.ReLU(inplace=True)
        self.fc6 = nn.Linear(1024, self.z_dim)

    def forward(self, x):
        """forward pass for encoder.
        x: img with (3, 64, 64)
        """
        b = x.size(0)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.dropout(out)
        out = out.view(b, -1)
        out = self.fc5(out)
        out = self.relu(out)
        return self.fc6(out)

class Decoder(nn.Module):
    def __init__(self, out_c=3, z_dim=512):
        super().__init__()
        self.out_c = out_c
        self.z_dim = z_dim

        self.fc1 = nn.Linear(z_dim, 1024)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(1024, 1280)

        self.convt3 = UpBlock(80, 40, kernel_size=3, stride=3, output_padding=1)
        self.convt4 = UpBlock(40, 20, kernel_size=5, stride=1, output_padding=0)
        self.convt5 = UpBlock(20, 10, kernel_size=7, stride=3, output_padding=1)
        #self.convt6 = UpBlock(10, out_c, kernel_size=9, stride=1, output_padding=0)
        self.convt6 = nn.ConvTranspose2d(10, out_c, kernel_size=9, stride=1, bias=False)
        self.tanh = nn.Tanh() # NOTE: Different from the VideoVAE paper

    def forward(self, z):
        out = self.fc1(z)
        out = self.relu(out)
        out = self.fc2(out)

        out = out.view(-1, 80, 4, 4) # the shape from encoder before its first FC layers
        out = self.convt3(out)
        out = self.convt4(out)
        out = self.convt5(out)
        out = self.convt6(out)
        return self.tanh(out)

class AttributeNet(nn.Module):
    def __init__(self, z_dim=512, h_dim=128, n_act=10, n_id=9):
        super().__init__()
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.n_act = n_act
        self.n_id = n_id

        self.fc1 = nn.Linear(z_dim, self.h_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc_act = nn.Linear(self.h_dim, n_act)
        self.fc_id = nn.Linear(self.h_dim, n_id)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        logit_act = self.fc_act(out)
        logit_id = self.fc_id(out)
        return logit_act, logit_id

class AttributeNet_v2(nn.Module):
    def __init__(self, z_dim=512, h_dim=128, n_act=10, n_id=9):
        super().__init__()
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.n_act = n_act
        self.n_id = n_id

        self.fc1_act = nn.Linear(z_dim, self.h_dim)
        self.fc1_id = nn.Linear(z_dim, self.h_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2_act = nn.Linear(self.h_dim, n_act)
        self.fc2_id = nn.Linear(self.h_dim, n_id)
    
    def forward(self, x):
        h1_act = self.relu(self.fc1_act(x))
        h1_id = self.relu(self.fc1_id(x))
        logit_act = self.fc2_act(h1_act)
        logit_id = self.fc2_id(h1_id)

        return logit_act, logit_id

# ref: https://github.com/pytorch/examples/blob/master/vae/main.py
class DistributionNet(nn.Module):
    def __init__(self, in_dim=512, h_dim=512, out_dim=512):
        super().__init__()

        self.in_dim = in_dim
        self.h_dim = h_dim
        self.out_dim = out_dim
        
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc21 = nn.Linear(h_dim, out_dim)
        self.fc22 = nn.Linear(h_dim, out_dim)

    # follow the term in VAE. check: https://github.com/pytorch/examples/blob/master/vae/main.py#L49-L51
    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        mu, logvar = self.fc21(h1), self.fc22(h1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu)
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

class MLP(nn.Module):
    def __init__(self, in_dim=512, h_dim=512, out_dim=512):
        super().__init__()
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.out_dim = out_dim

        self.fc1 = nn.Linear(in_dim, h_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(h_dim, out_dim)

    def forward(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc2(h1)

########################################
#                 Ours                 #
########################################

class ConditionalBatchNorm(nn.Module):
    def __init__(self, shape, n_features=3):
        super().__init__()
        self.n_features = n_features
        self.shape = shape
        self.eps = 1e-5

        n1 = np.prod(np.array(list(shape))) * n_features
        self.fc1 = nn.Linear(n1, 9)
        self.relu = nn.ReLU(inplace=True)
        self.fc2_beta = nn.Linear(9, self.n_features)
        self.fc2_gamma = nn.Linear(9, self.n_features)

    def forward(self, x):
        assert x.size(1) == self.n_features

        b = x.size(0)
        # get mu, std per channel
        #mu  = x.permute(1, 0, 2, 3).view(b, self.n_features, -1).mean(dim=2, keepdim=True)
        mu  = x.permute(1, 0, 2, 3).contiguous().view(self.n_features, -1).mean(dim=1)
        std = x.permute(1, 0, 2, 3).contiguous().view(self.n_features, -1).std(dim=1)
        #TODO BatchNorm
        #mu  = x.view(b, self.n_features, -1).mean(dim=2, keepdim=True)
        #std = x.view(b, self.n_features, -1).std(dim=2, keepdim=True)
        h1 = self.relu(self.fc1(x.view(b, -1)))
        beta = self.fc2_beta(h1)    # (b, n_features)
        gamma = self.fc2_gamma(h1)  # (b, n_features)
        
        out = (x - mu[None, :, None, None]) / (std[None, :, None, None] + self.eps) * gamma[:, :, None, None] + beta[:, :, None, None]

        return out

class ConvBNReLU(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride=1, padding=1, bias=False, norm='CBN', shape=None):
        super().__init__()

        self.conv = nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        if norm == 'CBN':
            self.bn = ConditionalBatchNorm(shape, n_features=out_c)
        else:
            self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out

# encode style, i.e. identity
class StyleEncoder(nn.Module):
    def __init__(self, in_c=3):
        super().__init__()

        self.conv1 = ConvBNReLU(3, 8, kernel_size=3, shape=(28, 28))
        self.conv2 = ConvBNReLU(8, 16, kernel_size=3, shape=(28, 28))
        self.conv3 = ConvBNReLU(16, 32, kernel_size=3, shape=(28, 28))

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        return out

# encode content, i.e. action
class ContentEncoder(nn.Module):
    def __init__(self, in_c=3):
        super().__init__()

        self.conv1 = ConvBNReLU(3, 8, kernel_size=3, shape=(28, 28))
        self.conv2 = ConvBNReLU(8, 16, kernel_size=3, shape=(28, 28))
        self.conv3 = ConvBNReLU(16, 32, kernel_size=3, shape=(28, 28))

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        return out

class OurDecoder(nn.Module):
    def __init__(self, in_c=64):
        super().__init__()

        self.convt1 = ConvBNReLU(3, 8, kernel_size=3, shape=(28, 28))
        self.convt2 = ConvBNReLU(8, 16, kernel_size=3, shape=(28, 28))
        self.convt3 = ConvBNReLU(16, 32, kernel_size=3, shape=(28, 28))

    def forward(self, x):
        out = self.convt1(x)
        out = self.convt2(out)
        out = self.convt3(out)
        return out

class MotionNet(nn.Module):
    def __init__(self):
        super().__init__()

        input_size = 512
        hidden_size = 128
        num_layers = 1
        bidirectional = False

        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            bidirectional=bidirectional)

    def forward(self, x):


        return x