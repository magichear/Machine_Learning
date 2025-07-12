import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_dice_loss(y_pred, y_true, epsilon=1e-6):
    """
    Soft dice loss calculation for arbitrary batch size, number of classes, and number of spatial dimensions.
    Assumes the `channels_last` format.

    # Arguments
        y_true: b x X x Y( x Z...) Need to transform to b x c x X x Y( x Z...) One hot encoding of ground truth
        y_pred: b x c x X x Y( x Z...) Network output, must sum to 1 over c channel (such as after softmax)
        epsilon: Used for numerical stability to avoid divide by zero errors

    # References
        V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation
        https://arxiv.org/abs/1606.04797
        More details on Dice loss formulation
        https://mediatum.ub.tum.de/doc/1395260/1395260.pdf (page 72)

        Adapted from https://github.com/Lasagne/Recipes/issues/99#issuecomment-347775022
    """

    # skip the batch and class axis for calculating Dice score
    y_true = torch.unsqueeze(y_true, dim=1)
    y_pred = F.softmax(y_pred, dim=1)
    y_pred = y_pred[:, 1:, :, :]
    axes = tuple(range(2, len(y_pred.shape)))
    numerator = 2.0 * torch.sum(y_pred * y_true, axes)
    denominator = torch.sum(torch.square(y_pred) + torch.square(y_true), axes)

    return 1 - torch.mean(
        (numerator + 1.0) / (denominator + epsilon + 1.0)
    )  # average over classes and batch


class SiLU(nn.Module):
    """export-friendly version of nn.SiLU()"""

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(
        self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"
    ):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class DWConv(nn.Module):
    """Depthwise Conv + Conv"""

    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(
            in_channels,
            in_channels,
            ksize=ksize,
            stride=stride,
            groups=in_channels,
            act=act,
        )
        self.pconv = BaseConv(
            in_channels, out_channels, ksize=1, stride=1, groups=1, act=act
        )

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


def double_convolution(in_channels, out_channels):
    """
    In the original paper implementation, the convolution operations were
    not padded but we are padding them here. This is because, we need the
    output result size to be same as input size.
    """
    """conv_op = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )"""
    conv_op = Bottleneck(in_channels, out_channels)
    return conv_op


class UNet5(nn.Module):
    def __init__(self, num_classes):
        super(UNet5, self).__init__()

        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)

        # Contracting path.
        # Each convolution is applied twice.
        self.down_convolution_1 = double_convolution(3, 64)
        self.down_convolution_2 = double_convolution(64, 128)
        self.down_convolution_3 = double_convolution(128, 256)
        self.down_convolution_4 = double_convolution(256, 512)
        self.down_convolution_5 = double_convolution(512, 1024)

        # Expanding path.
        self.up_transpose_1 = nn.ConvTranspose2d(
            in_channels=1024, out_channels=512, kernel_size=2, stride=2
        )
        self.up_convolution_1 = double_convolution(1024, 512)
        self.up_transpose_2 = nn.ConvTranspose2d(
            in_channels=512, out_channels=256, kernel_size=2, stride=2
        )
        self.up_convolution_2 = double_convolution(512, 256)
        self.up_transpose_3 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128, kernel_size=2, stride=2
        )
        self.up_convolution_3 = double_convolution(256, 128)
        self.up_transpose_4 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=2, stride=2
        )
        self.up_convolution_4 = double_convolution(128, 64)

        # output => increase the `out_channels` as per the number of classes.
        self.out = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        down_1 = self.down_convolution_1(x)
        down_2 = self.max_pool2d(down_1)  # 1/2
        down_3 = self.down_convolution_2(down_2)
        down_4 = self.max_pool2d(down_3)  # 1/4
        down_5 = self.down_convolution_3(down_4)
        down_6 = self.max_pool2d(down_5)  # 1/8
        down_7 = self.down_convolution_4(down_6)
        down_8 = self.max_pool2d(down_7)  # 1/16
        down_9 = self.down_convolution_5(down_8)

        up_1 = self.up_transpose_1(down_9)  # 1/8
        x = self.up_convolution_1(torch.cat([down_7, up_1], 1))

        up_2 = self.up_transpose_2(x)  # 1/4
        x = self.up_convolution_2(torch.cat([down_5, up_2], 1))

        up_3 = self.up_transpose_3(x)  # 1/2
        x = self.up_convolution_3(torch.cat([down_3, up_3], 1))

        up_4 = self.up_transpose_4(x)  # 1/1
        x = self.up_convolution_4(torch.cat([down_1, up_4], 1))

        out = self.out(x)
        return out


class UNetPP(nn.Module):
    def __init__(self, num_classes):
        super(UNetPP, self).__init__()

        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)

        # Contracting path
        self.down_convolution_1 = double_convolution(3, 64)
        self.down_convolution_2 = double_convolution(64, 128)
        self.down_convolution_3 = double_convolution(128, 256)
        self.down_convolution_4 = double_convolution(256, 512)
        self.down_convolution_5 = double_convolution(512, 1024)

        # Expanding path with nested skip connections
        self.up_transpose_1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_convolution_1_0 = double_convolution(1024, 512)
        self.up_convolution_1_1 = double_convolution(1024, 512)

        self.up_transpose_2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_convolution_2_0 = double_convolution(512, 256)
        self.up_convolution_2_1 = double_convolution(512, 256)
        self.up_convolution_2_2 = double_convolution(512, 256)

        self.up_transpose_3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_convolution_3_0 = double_convolution(256, 128)
        self.up_convolution_3_1 = double_convolution(256, 128)
        self.up_convolution_3_2 = double_convolution(256, 128)
        self.up_convolution_3_3 = double_convolution(256, 128)

        self.up_transpose_4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_convolution_4_0 = double_convolution(128, 64)
        self.up_convolution_4_1 = double_convolution(128, 64)
        self.up_convolution_4_2 = double_convolution(128, 64)
        self.up_convolution_4_3 = double_convolution(128, 64)
        self.up_convolution_4_4 = double_convolution(128, 64)

        # Output layer
        self.out = nn.Conv2d(64, num_classes, kernel_size=1)

        # Additional 1x1 convolution layers to reduce channels
        self.reduce_channels_x2_2 = nn.Conv2d(768, 512, kernel_size=1)
        self.reduce_channels_x1_2 = nn.Conv2d(384, 256, kernel_size=1)
        self.reduce_channels_x0_2 = nn.Conv2d(192, 128, kernel_size=1)
        self.reduce_channels_x1_3 = nn.Conv2d(512, 256, kernel_size=1)
        self.reduce_channels_x0_3 = nn.Conv2d(256, 128, kernel_size=1)
        self.reduce_channels_x0_4 = nn.Conv2d(
            256, 128, kernel_size=1
        )  # 修改输入通道数为 256

    def forward(self, x):
        # Contracting path
        x0_0 = self.down_convolution_1(x)
        x1_0 = self.down_convolution_2(self.max_pool2d(x0_0))
        x2_0 = self.down_convolution_3(self.max_pool2d(x1_0))
        x3_0 = self.down_convolution_4(self.max_pool2d(x2_0))
        x4_0 = self.down_convolution_5(self.max_pool2d(x3_0))

        # Expanding path with nested skip connections
        x3_1 = self.up_convolution_1_0(
            torch.cat([x3_0, self.up_transpose_1(x4_0)], dim=1)
        )
        x2_1 = self.up_convolution_2_0(
            torch.cat([x2_0, self.up_transpose_2(x3_1)], dim=1)
        )
        x1_1 = self.up_convolution_3_0(
            torch.cat([x1_0, self.up_transpose_3(x2_1)], dim=1)
        )
        x0_1 = self.up_convolution_4_0(
            torch.cat([x0_0, self.up_transpose_4(x1_1)], dim=1)
        )

        x2_2_input = torch.cat([x2_0, x2_1, self.up_transpose_2(x3_1)], dim=1)
        x2_2_input = self.reduce_channels_x2_2(x2_2_input)
        x2_2 = self.up_convolution_2_1(x2_2_input)

        x1_2_input = torch.cat([x1_0, x1_1, self.up_transpose_3(x2_2)], dim=1)
        x1_2_input = self.reduce_channels_x1_2(x1_2_input)
        x1_2 = self.up_convolution_3_1(x1_2_input)

        x1_3_input = torch.cat([x1_0, x1_1, x1_2, self.up_transpose_3(x2_2)], dim=1)
        x1_3_input = self.reduce_channels_x1_3(x1_3_input)
        x1_3 = self.up_convolution_3_2(x1_3_input)

        x0_2_input = torch.cat([x0_0, x0_1, self.up_transpose_4(x1_2)], dim=1)
        x0_2_input = self.reduce_channels_x0_2(x0_2_input)
        x0_2 = self.up_convolution_4_1(x0_2_input)

        x0_3_input = torch.cat([x0_0, x0_1, x0_2, self.up_transpose_4(x1_3)], dim=1)
        x0_3_input = self.reduce_channels_x0_3(x0_3_input)
        x0_3 = self.up_convolution_4_2(x0_3_input)

        x0_4_input = torch.cat([x0_0, x0_1, x0_2, x0_3], dim=1)
        x0_4_input = self.reduce_channels_x0_4(x0_4_input)
        x0_4 = self.up_convolution_4_3(x0_4_input)

        # Output layer
        out = self.out(x0_4)
        return out


class UNet3(nn.Module):
    def __init__(self, num_classes):
        super(UNet3, self).__init__()

        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)

        # Contracting path.
        # Each convolution is applied twice.
        self.down_convolution_1 = double_convolution(3, 64)
        self.down_convolution_2 = double_convolution(64, 128)
        self.down_convolution_3 = double_convolution(128, 256)

        # Expanding path.
        self.up_transpose_3 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128, kernel_size=2, stride=2
        )
        self.up_convolution_3 = double_convolution(256, 128)
        self.up_transpose_4 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=2, stride=2
        )
        self.up_convolution_4 = double_convolution(128, 64)

        # output => increase the `out_channels` as per the number of classes.
        self.out = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        down_1 = self.down_convolution_1(x)
        down_2 = self.max_pool2d(down_1)  # 1/2
        down_3 = self.down_convolution_2(down_2)
        down_4 = self.max_pool2d(down_3)  # 1/4
        down_5 = self.down_convolution_3(down_4)

        up_3 = self.up_transpose_3(down_5)  # 1/2
        x = self.up_convolution_3(torch.cat([down_3, up_3], 1))

        up_4 = self.up_transpose_4(x)  # 1/1
        x = self.up_convolution_4(torch.cat([down_1, up_4], 1))

        out = self.out(x)
        return out


if __name__ == "__main__":
    input_image = torch.rand((1, 3, 512, 512))
    model = UNet3(num_classes=10)
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"{total_trainable_params:,} training parameters.")
    outputs = model(input_image)
    print(outputs.shape)
