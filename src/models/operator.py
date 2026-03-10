import torch
import torch.nn as nn
from configuration.config import config

class ReLUConvBN(nn.Module):
    def __init__(self, C_in: int, C_out: int, kernel_size: int, stride: int, padding: int, affine: bool = True):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace= False),
            nn.Conv2d(C_in, C_out, kernel_size= kernel_size, stride= stride, padding= padding, bias= False),
            nn.BatchNorm2d(C_out, affine= affine)
        )
    
    def forward(self, x):
        return self.op(x)
    
class SepConv(nn.Module):
    def __init__(self, C_in: int, C_out: int, kernel_size: int, stride: int, padding: int, affine: bool = True):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace= False),
            nn.Conv2d(C_in, C_in, kernel_size= kernel_size, stride= stride, padding= padding, bias= False, groups= C_in),
            nn.Conv2d(C_in, C_in, kernel_size= 1, bias= False),
            nn.BatchNorm2d(C_in, affine= affine),
            nn.ReLU(inplace= False),
            nn.Conv2d(C_in, C_in, kernel_size= kernel_size, stride= 1, padding= padding, bias= False, groups= C_in),
            nn.Conv2d(C_in, C_out, kernel_size= 1, bias= False),
            nn.BatchNorm2d(C_out, affine= affine)
        )
        
    def forward(self, x):
        return self.op(x)
    
class DilConv(nn.Module):
    def __init__(self, C_in: int, C_out: int, kernel_size: int, stride: int, padding: int, dilation: int, affine: bool = True):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace= False),
            nn.Conv2d(C_in, C_in, kernel_size= kernel_size, stride= stride, padding= padding, dilation= dilation, groups= C_in, bias= False),
            nn.Conv2d(C_in, C_out, kernel_size= 1,bias= False),
            nn.BatchNorm2d(C_out, affine= affine)
        )
    
    def forward(self, x):
        return self.op(x)
    
class Identity(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x
    
class Zero(nn.Module):
    def __init__(self, stride: int):
        super().__init__()
        self.stride = stride
    
    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        else:
            return x[:, :, ::self.stride, ::self.stride].mul(0.)

class FactorizedReduce(nn.Module):
    def __init__(self, C_in: int, C_out: int, affine: bool = True):
        super().__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace= False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, kernel_size= 1, stride= 2, padding= 0, bias= False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, kernel_size= 1, stride= 2, padding= 0, bias= False)
        self.bn = nn.BatchNorm2d(C_out, affine= affine)
        
    def forward(self, x):
        x = self.relu(x)
        if x.size(2) <= 1 or x.size(3) <= 1:
            out = torch.cat([self.conv_1(x), self.conv_1(x)], dim=1)
        else:
            out1 = self.conv_1(x)
            out2 = self.conv_2(x[:, :, 1:, 1:])
            # 处理奇数分辨率: 当 H 或 W 为奇数时, stride=2 的输出尺寸可能不一致
            if out1.size() != out2.size():
                out2 = torch.nn.functional.pad(out2, [0, out1.size(3) - out2.size(3), 0, out1.size(2) - out2.size(2)])
            out = torch.cat([out1, out2], dim=1)
        out = self.bn(out)
        
        return out
    
class DropPath(nn.Module):
    """Scheduled DropPath (Stochastic Depth): drops entire sample paths during training."""
    def __init__(self):
        super().__init__()
        self.drop_prob = 0.0

    def forward(self, x):
        if not self.training or self.drop_prob <= 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        mask = torch.zeros(x.shape[0], 1, 1, 1, device=x.device, dtype=x.dtype).bernoulli_(keep_prob)
        return x * mask / keep_prob

def get_op(op_name: str, C: int, stride: int, affine: bool = True) -> nn.Module:
    OPS = {
        'zero': lambda C, stride, affine: Zero(stride),
        'skip_connect': lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
        'sep_conv_3x3':lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine= affine),
        'sep_conv_5x5':lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine= affine),
        'sep_conv_7x7':lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine= affine),
        'dil_conv_3x3':lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine= affine),
        'dil_conv_5x5':lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine= affine),
        'conv_7x1_1x7':lambda C, stride, affine: nn.Sequential(
            nn.ReLU(inplace= False),
            nn.Conv2d(C, C, (1, 7), stride= (1,stride), padding= (0, 3), bias= False),
            nn.Conv2d(C, C, (7, 1), stride= (stride, 1), padding= (3, 0), bias= False),
            nn.BatchNorm2d(C, affine= affine)
        ),        
        'avg_pool_3x3':lambda C, stride, affine: nn.AvgPool2d(3, stride= stride, padding= 1, count_include_pad= False),
        'max_pool_3x3':lambda C, stride, affine: nn.MaxPool2d(3, stride= stride, padding= 1)
    }
    
    if op_name not in OPS:
        raise ValueError(f'Unkown operation: {op_name}')
    
    return OPS[op_name](C, stride, affine)

