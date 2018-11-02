from torch import nn

class Conv2dSame(nn.Module):
    """ Applies a 2D convolution  over an input signal composed of several input
        planes and output size is the same as input size.

        Args:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (only tuple): Size of the convolving kernel
            stride (only int): Stride of the convolution. Default: 1
            padding_layer: type of layer for padding. Default: nn.ZeroPad2d
            dilation (only int): Spacing between kernel elements. Default: 1
            bias (bool): If ``True``, adds a learnable bias to the output. Default: ``True``
        
        Shape:
            - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
            - Output: :math:`(N, C_{out}, H_{out}, W_{out})` where
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding_layer=nn.ZeroPad2d, dilation=1, bias=True):
        super().__init__()
        
        kernel_h, kernel_w = kernel_size
        
        k = dilation*(kernel_w-1)-stride+2
        pr = k // 2
        pl = pr - 1 if k % 2 == 0 else pr
        
        k = dilation*(kernel_h-1)-stride+1+1
        pb = k // 2
        pt = pb - 1 if k % 2 == 0 else pb
        self.pad_same = padding_layer((pl, pr, pb, pt))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                              stride, dilation=dilation, bias=bias)

    def forward(self, x):
        x = self.pad_same(x)
        x = self.conv(x)
        return x
