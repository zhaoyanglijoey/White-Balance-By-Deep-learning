import torch
import torch.nn as nn


class TransformerNet(nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()
        self.conv1 = ConvLayer(in_channels=3, out_channels=32, kernel_size=9, stride=1)
        self.ins1 = nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer(32, 64, 3, 2)
        self.ins2 = nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvLayer(64, 128, 3, 2)
        self.ins3 = nn.InstanceNorm2d(128, affine=True)

        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)

        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.ins4 = nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.ins5 = nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()

    def forward(self, x):
        out = self.relu(self.ins1(self.conv1(x)))
        out = self.relu(self.ins2(self.conv2(out)))
        out = self.relu(self.ins3(self.conv3(out)))
        out = self.res1(out)
        out = self.res2(out)
        out = self.res3(out)
        out = self.res4(out)
        out = self.res5(out)
        out = self.relu(self.ins4(self.deconv1(out)))
        out = self.relu(self.ins5(self.deconv2(out)))
        out = self.sigmoid(self.deconv3(out))
        return out

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = ConvLayer(in_channels=3, out_channels=32, kernel_size=9, stride=1) # 256
        self.ins1 = nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer(32, 64, 3, 2) # 128
        self.ins2 = nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvLayer(64, 128, 3, 2) # 64
        self.ins3 = nn.InstanceNorm2d(128, affine=True)

        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)

        self.conv4 = ConvLayer(128, 64, kernel_size=3, stride=2) # 32
        self.ins4 = nn.InstanceNorm2d(64, affine=True)
        self.conv5 = ConvLayer(64, 32, kernel_size=3, stride=2) # 16
        self.ins5 = nn.InstanceNorm2d(32, affine=True)
        self.conv6 = ConvLayer(32, 16, kernel_size=3, stride=2) # 8
        self.ins6 = nn.InstanceNorm2d(16, affine=True)
        self.conv7 = nn.Conv2d(16, 1, kernel_size=8, stride=1)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()
        self.lrelu = nn.LeakyReLU()

    def forward(self, x):
        out = self.lrelu(self.ins1(self.conv1(x)))
        out = self.lrelu(self.ins2(self.conv2(out)))
        out = self.lrelu(self.ins3(self.conv3(out)))
        out = self.res1(out)
        out = self.res2(out)
        out = self.res3(out)
        out = self.res4(out)
        out = self.res5(out)
        out = self.lrelu(self.ins4(self.conv4(out)))
        out = self.lrelu(self.ins5(self.conv5(out)))
        out = self.lrelu(self.ins6(self.conv6(out)))
        out = self.sigmoid(self.conv7(out))
        out = out.view(-1)
        return out

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        pad_size = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(pad_size)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv(out)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, 3, 1)
        self.ins1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, 3, 1)
        self.ins2 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.ins1(self.conv1(x)))
        out = self.ins2(self.conv2(out))
        out = out + residual
        return out


class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample_layer = nn.Upsample(mode='nearest', scale_factor=upsample)
        pad_size = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(pad_size)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = self.upsample_layer(x_in)
        out = self.reflection_pad(x_in)
        out = self.conv(out)
        return out