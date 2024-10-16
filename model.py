import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, num_filters):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return x

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, num_filters):
        super(EncoderBlock, self).__init__()
        self.conv_block = ConvBlock(in_channels, num_filters)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv_block(x)
        p = self.pool(x)
        return x, p

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, num_filters):
        super(DecoderBlock, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, num_filters, kernel_size=2, stride=2)
        self.conv_block = ConvBlock(in_channels, num_filters)

    def forward(self, x, skip_features):
        x = self.conv_transpose(x)
        x = torch.cat([x, skip_features], axis=1)  # Concatenate along the channel dimension
        x = self.conv_block(x)
        return x

class UNet(nn.Module):
    def __init__(self, num_classes, in_channels=3):  # Default to 3 channels (RGB)
        super(UNet, self).__init__()

        self.encoder1 = EncoderBlock(in_channels, 64)   # Pass in_channels
        self.encoder2 = EncoderBlock(64, 128)
        self.encoder3 = EncoderBlock(128, 256)
        self.encoder4 = EncoderBlock(256, 512)
        self.bottleneck = ConvBlock(512, 1024)

        self.decoder1 = DecoderBlock(1024, 512)  # 1024 from bottleneck
        self.decoder2 = DecoderBlock(512, 256)
        self.decoder3 = DecoderBlock(256, 128)
        self.decoder4 = DecoderBlock(128, 64)
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1, padding=0)

    def forward(self, x):
        s1, p1 = self.encoder1(x)
        s2, p2 = self.encoder2(p1)
        s3, p3 = self.encoder3(p2)
        s4, p4 = self.encoder4(p3)

        b1 = self.bottleneck(p4)  # Use bottleneck layer

        d1 = self.decoder1(b1, s4)
        d2 = self.decoder2(d1, s3)
        d3 = self.decoder3(d2, s2)
        d4 = self.decoder4(d3, s1)

        outputs = self.final_conv(d4)  # Correct reference to final output
        return outputs

