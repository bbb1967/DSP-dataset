import torch.nn as nn  # nn模块包含了用于构建神经网络的类和函数
import torch.nn.functional as F  # 这行代码导入PyTorch深度学习库的函数模块，其中F模块包含了一些常用的神经网络函数，如激活函数、损失函数等
import torch
from torch.nn import init  # 从PyTorch的神经网络模块中导入参数初始化的函数
from einops import rearrange
from torchinfo import summary
from timm.models.layers import DropPath, to_2tuple
from torch import nn, einsum
from sklearn.decomposition import PCA
from typing import Union, Tuple, Optional
import torch
from einops import rearrange
from torch import Tensor
NUM_CLASS = 11

class Swish(nn.Module):
    def __init__(self) -> None:
        super(Swish, self).__init__()

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs * inputs.sigmoid()


class GLU(nn.Module):
    def __init__(self, dim: int) -> None:
        super(GLU, self).__init__()
        self.dim = dim

    def forward(self, inputs: Tensor) -> Tensor:
        outputs, gate = inputs.chunk(2, dim=self.dim)
        return outputs * gate.sigmoid()


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, inputs: Tensor) -> Tensor:
        return (inputs * torch.tanh(F.softplus(inputs)))


class MultiConvModule(nn.Module):
    """
    The extension from ConvModule1 with multiscale dilation for 4D input
    """

    def __init__(
            self,
            in_channels: int,
            kernel_size: int = 31,
            expansion_factor: int = 2,
            dropout_p: float = 0.1,
            dilation_rate: int = 1
    ) -> None:
        super(MultiConvModule, self).__init__()
        assert (kernel_size - 1) % 2 == 0, "kernel_size should be an odd number for 'SAME' padding"

        out_channels = in_channels * expansion_factor
        i = 2 ** dilation_rate
        kernel_size2 = kernel_size + 2

        padding_size1 = ((kernel_size - 1) * (i - 1) + kernel_size - 1) // 2
        padding_size2 = ((kernel_size2 - 1) * (i - 1) + kernel_size2 - 1) // 2

        self.c1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU()
        self.c2 = nn.Conv2d(out_channels, out_channels, kernel_size, groups=out_channels, stride=1, dilation=i,
                            padding=padding_size1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.swish = Swish()
        self.c3 = nn.Conv2d(out_channels, out_channels, kernel_size2, groups=out_channels, stride=1, dilation=i,
                            padding=padding_size2)
        self.c4 = nn.Conv2d(out_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.do = nn.Dropout2d(p=dropout_p)

    def forward(self, x):
        x = self.c1(x)
        x = self.relu(x)
        x1 = self.c2(x)
        x1 = self.bn2(x1)
        x1 = self.swish(x1)
        x2 = self.c3(x)
        x2 = self.bn2(x2)
        x2 = self.swish(x2)
        x = x1 + x2
        x = self.c4(x)
        x = self.do(x)

        return x
class Involution2d(nn.Module):


    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 sigma_mapping: Optional[nn.Module] = None,
                 kernel_size: Union[int, Tuple[int, int]] = (3, 3),
                 stride: Union[int, Tuple[int, int]] = (1, 1),
                 groups: int = 1,
                 reduce_ratio: int = 1,
                 dilation: Union[int, Tuple[int, int]] = (1, 1),
                 padding: Union[int, Tuple[int, int]] = (3, 3),
                 bias: bool = False,
                 **kwargs) -> None:
        """
        Constructor method
        :param in_channels: (int) Number of input channels
        :param out_channels: (int) Number of output channels
        :param sigma_mapping: (nn.Module) Non-linear mapping as introduced in the paper. If none BN + ReLU is utilized
        :param kernel_size: (Union[int, Tuple[int, int]]) Kernel size to be used
        :param stride: (Union[int, Tuple[int, int]]) Stride factor to be utilized
        :param groups: (int) Number of groups to be employed
        :param reduce_ratio: (int) Reduce ration of involution channels
        :param dilation: (Union[int, Tuple[int, int]]) Dilation in unfold to be employed
        :param padding: (Union[int, Tuple[int, int]]) Padding to be used in unfold operation
        :param bias: (bool) If true bias is utilized in each convolution layer
        :param **kwargs: Unused additional key word arguments
        """
        # Call super constructor
        super(Involution2d, self).__init__()
        # Check parameters
        assert isinstance(in_channels, int) and in_channels > 0, "in channels must be a positive integer."
        assert in_channels % groups == 0, "out_channels must be divisible by groups"
        assert isinstance(out_channels, int) and out_channels > 0, "out channels must be a positive integer."
        assert out_channels % groups == 0, "out_channels must be divisible by groups"
        assert isinstance(sigma_mapping, nn.Module) or sigma_mapping is None, \
            "Sigma mapping must be an nn.Module or None to utilize the default mapping (BN + ReLU)."
        assert isinstance(kernel_size, int) or isinstance(kernel_size, tuple), \
            "kernel size must be an int or a tuple of ints."
        assert isinstance(stride, int) or isinstance(stride, tuple), \
            "stride must be an int or a tuple of ints."
        assert isinstance(groups, int), "groups must be a positive integer."
        assert isinstance(reduce_ratio, int) and reduce_ratio > 0, "reduce ratio must be a positive integer."
        assert isinstance(dilation, int) or isinstance(dilation, tuple), \
            "dilation must be an int or a tuple of ints."
        assert isinstance(padding, int) or isinstance(padding, tuple), \
            "padding must be an int or a tuple of ints."
        assert isinstance(bias, bool), "bias must be a bool"
        # Save parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.groups = groups
        self.reduce_ratio = reduce_ratio
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.bias = bias
        # Init modules
        self.sigma_mapping = sigma_mapping if sigma_mapping is not None else nn.Sequential(
            nn.BatchNorm2d(num_features=self.out_channels // self.reduce_ratio, momentum=0.3), nn.ReLU())
        self.initial_mapping = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                         kernel_size=(1, 1), stride=(1, 1), padding=(0, 0),
                                         bias=bias) if self.in_channels != self.out_channels else nn.Identity()
        self.o_mapping = nn.AvgPool2d(kernel_size=self.stride, stride=self.stride)
        self.reduce_mapping = nn.Conv2d(in_channels=self.in_channels,
                                        out_channels=self.out_channels // self.reduce_ratio, kernel_size=(1, 1),
                                        stride=(1, 1), padding=(0, 0), bias=bias)
        self.span_mapping = nn.Conv2d(in_channels=self.out_channels // self.reduce_ratio,
                                      out_channels=self.kernel_size[0] * self.kernel_size[1] * self.groups,
                                      kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=bias)
        self.unfold = nn.Unfold(kernel_size=self.kernel_size, dilation=dilation, padding=padding, stride=stride)

    def __repr__(self) -> str:
        """
        Method returns information about the module
        :return: (str) Info string
        """
        return ("{}({}, {}, kernel_size=({}, {}), stride=({}, {}), padding=({}, {}), "
                "groups={}, reduce_ratio={}, dilation=({}, {}), bias={}, sigma_mapping={})".format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels,
            self.kernel_size[0],
            self.kernel_size[1],
            self.stride[0],
            self.stride[1],
            self.padding[0],
            self.padding[1],
            self.groups,
            self.reduce_mapping,
            self.dilation[0],
            self.dilation[1],
            self.bias,
            str(self.sigma_mapping)
        ))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param input: (torch.Tensor) Input tensor of the shape [batch size, in channels, height, width]
        :return: (torch.Tensor) Output tensor of the shape [batch size, out channels, height, width] (w/ same padding)
        """
        # Check input dimension of input tensor
        assert input.ndimension() == 4, \
            "Input tensor to involution must be 4d but {}d tensor is given".format(input.ndimension())
        # Save input shape and compute output shapes
        batch_size, _, in_height, in_width = input.shape
        out_height = (in_height + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) \
                     // self.stride[0] + 1
        out_width = (in_width + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) \
                    // self.stride[1] + 1
        # Unfold and reshape input tensor
        input_unfolded = self.unfold(self.initial_mapping(input))
        input_unfolded = input_unfolded.view(batch_size, self.groups, self.out_channels // self.groups,
                                             self.kernel_size[0] * self.kernel_size[1],
                                             out_height, out_width)
        # Generate kernel
        kernel = self.span_mapping(self.sigma_mapping(self.reduce_mapping(self.o_mapping(input))))
        kernel = kernel.view(batch_size, self.groups, self.kernel_size[0] * self.kernel_size[1],
                             kernel.shape[-2], kernel.shape[-1]).unsqueeze(dim=2)
        # Apply kernel to produce output
        output = (kernel * input_unfolded).sum(dim=3)
        # Reshape output
        output = output.view(batch_size, -1, output.shape[-2], output.shape[-1])
        return output


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 8, drop_p: float = 0.):
        print ("expansion: ", expansion)
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            #Swish(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
            nn.Dropout(drop_p)
        )



class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class FusionConv(nn.Module):
    def __init__(self, in_channels, out_channels, factor=4.0):
        super(FusionConv, self).__init__()
        dim = int(out_channels // factor)
        self.down = nn.Conv2d(in_channels, dim, kernel_size=1, stride=1)
        self.conv_3x3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.conv_5x5 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2)
        self.conv_7x7 = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3)
        self.spatial_attention = SpatialAttentionModule()
        self.channel_attention = ChannelAttentionModule(dim)
        self.up = nn.Conv2d(dim, out_channels, kernel_size=1, stride=1)
        self.down_2 = nn.Conv2d(in_channels, dim, kernel_size=1, stride=1)

    def forward(self, x1, x2, x4):
        x_fused = torch.cat([x1, x2, x4], dim=1)
        x_fused = self.down(x_fused)
        x_fused_c = x_fused * self.channel_attention(x_fused)
        x_3x3 = self.conv_3x3(x_fused)
        x_5x5 = self.conv_5x5(x_fused)
        x_7x7 = self.conv_7x7(x_fused)
        x_fused_s = x_3x3 + x_5x5 + x_7x7
        x_fused_s = x_fused_s * self.spatial_attention(x_fused_s)

        x_out = self.up(x_fused_s + x_fused_c)

        return x_out

class MSAA(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MSAA, self).__init__()
        self.fusion_conv = FusionConv(in_channels, out_channels)

    def forward(self, x1, x2, x4, last=False):
        # # x2 是从低到高，x4是从高到低的设计，x2传递语义信息，x4传递边缘问题特征补充
        # x_1_2_fusion = self.fusion_1x2(x1, x2)
        # x_1_4_fusion = self.fusion_1x4(x1, x4)
        # x_fused = x_1_2_fusion + x_1_4_fusion
        x_fused = self.fusion_conv(x1, x2, x4)
        return x_fused

class CrossAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 8, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_k = nn.Linear(dim, inner_dim , bias=False)
        self.to_v = nn.Linear(dim, inner_dim , bias = False)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x_qkv):
        b, n, _, h = *x_qkv.shape, self.heads

        k = self.to_k(x_qkv)
        k = rearrange(k, 'b n (h d) -> b h n d', h = h)

        v = self.to_v(x_qkv)
        v = rearrange(v, 'b n (h d) -> b h n d', h = h)

        q = self.to_q(x_qkv[:, 0].unsqueeze(1))
        q = rearrange(q, 'b n (h d) -> b h n d', h = h)



        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out


class DSP2Net(nn.Module):  # Tri-CNN 构建的三分支3DCNN网络模型
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d):
            init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)

    def __init__(self, n_classes=9, patch_size=9, input_channels=1):
        super(DSP2Net, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size

        self.conv3d_spatial = nn.Sequential(
            nn.Conv3d(input_channels, out_channels=64, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(64),
            nn.Conv3d(64, out_channels=8, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.ReLU(),

        )
        self.conv3d_spectral = nn.Sequential(
            nn.Conv3d(input_channels, out_channels=64, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(64),
            nn.Conv3d(64, out_channels=8, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.ReLU(),

        )
        self.conv3d_spa_ape = nn.Sequential(
            nn.Conv3d(input_channels, out_channels=64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.Conv3d(64, out_channels=8, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),

        )
        self.conv2d_features_1 = nn.Sequential(
            Involution2d(in_channels=136, out_channels=128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            #Second_feature_filtering_att(channel=128)  # 添加空间注意力分支
        )
        self.nn1 = nn.Linear(128, n_classes)
        self.feed_forward = FeedForwardBlock(emb_size=128)
        self.apply(self.weight_init)
        self.bn = nn.BatchNorm1d(num_features=128)
        self.layer_norm = nn.LayerNorm(128)
        self.msaa = MSAA(in_channels=144, out_channels=136)
        self.multi_conv = MultiConvModule(in_channels=136)
        #self.multi_conv1 = MultiConvModule(in_channels=16)
        self.cross_attention = CrossAttention(dim=128, heads=8, dim_head=8, dropout=0.1)
    def forward(self, x):  # forward方法，它定义了数据在神经网络中的前向传播过程
        x_1 = self.conv3d_spatial(x)
        #print("Shape of x_1:", x_1.shape)
        x_2 = self.conv3d_spectral(x)
        x_3 = self.conv3d_spa_ape(x)

        #x = torch.cat([x_1, x_2, x_3], dim=1)
        #print(x.shape)
        # 去除光谱维度
        x_1 = rearrange(x_1, 'b c h w y -> b (c h) w y')
        x_2 = rearrange(x_2, 'b c h w y -> b (c h) w y')
        x_3 = rearrange(x_3, 'b c h w y -> b (c h) w y')
        #print(x.shape)
        x_msaa = self.msaa(x_1, x_2, x_3)
        #x = self.conv2d_features_1(x_msaa)
        x = self.multi_conv(x_msaa)
        x = self.conv2d_features_1(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        
        x = self.cross_attention(x)

        x = x.mean(dim=1)
        # 进行平均池化操作
        x = F.gelu(x)
        #print(x.shape)
        x = self.feed_forward(x)
        x = self.layer_norm(x)
        x = self.nn1(x)
        return x


if __name__ == '__main__':
    model = DSP2Net()
    # model.eval()   # 模型切换到推理模式
    # print(model)
    input = torch.randn(64, 1, 6, 9, 9)
    print("input shape:", input.shape)
    # y = model(input)
    # print(y.size())
    #summary(model, input_size=input.shape, device="cpu")
    print("output shape:", model(input).shape)
