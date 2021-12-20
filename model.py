import torch.nn as nn
import torch.nn.functional as F
import torch

from torchsummary import summary


# Output Size = (W - F + 2P) / S + 1
#
# W: input_volume_size
# F: kernel_size
# P: padding_size
# S: strides

class DeepSVDDNetwork(nn.Module):
    def __init__(self, z_dim=32):
        super(DeepSVDDNetwork, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 8, 5, bias=False, padding=2)
        self.bn1 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(8, 4, 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(4 * 16 * 16, 2 * 16 * 16, bias=False)
        self.fc2 = nn.Linear(2 * 16 * 16, z_dim, bias=False)

    def forward(self, x):
        # [3,64, 64]
        x = self.conv1(x)
        # [8, 32, 32]
        x = self.pool(F.leaky_relu(self.bn1(x)))
        # [8, 32, 32]
        x = self.conv2(x)
        # [4, 16, 16]
        x = self.pool(F.leaky_relu(self.bn2(x)))
        # [4, 8, 8]
        x = x.view(x.size(0), -1)
        # [-1, 1024]
        x = self.fc1(x)
        # [-1, 512]
        return self.fc2(x)
        # [-1, 32]


class pretrain_autoencoder(nn.Module):
    def __init__(self, z_dim=32):
        super(pretrain_autoencoder, self).__init__()
        self.z_dim = z_dim
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 8, 5, bias=False, padding=2)
        self.bn1 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(8, 4, 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(4 * 16 * 16, 2 * 16 * 16, bias=False)
        self.fc2 = nn.Linear(2 * 16 * 16, z_dim, bias=False)

        self.deconv1 = nn.ConvTranspose2d(2, 4, 5, bias=False, padding=2)
        self.bn3 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose2d(4, 8, 5, bias=False, padding=2)
        self.bn4 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.deconv3 = nn.ConvTranspose2d(8, 5, 5, bias=False, padding=2)
        self.bn5 = nn.BatchNorm2d(5, eps=1e-04, affine=False)
        self.deconv4 = nn.ConvTranspose2d(5, 3, 5, bias=False, padding=2)

    def encoder(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return self.fc2(x)

    def decoder(self, x):
        x = x.view(x.size(0), int(self.z_dim / 16), 4, 4)
        x = F.interpolate(F.leaky_relu(x), scale_factor=2)
        x = self.deconv1(x)
        x = F.interpolate(F.leaky_relu(self.bn3(x)), scale_factor=2)
        x = self.deconv2(x)
        x = F.interpolate(F.leaky_relu(self.bn4(x)), scale_factor=2)
        x = self.deconv3(x)
        x = F.interpolate(F.leaky_relu(self.bn5(x)), scale_factor=2)
        x = self.deconv4(x)
        return torch.sigmoid(x)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat


# if __name__ == "__main__":
#     model = pretrain_autoencoder().cuda()
#     # model = DeepSVDDNetwork().cuda()
#     summary(model, input_size=(3, 64, 64))
