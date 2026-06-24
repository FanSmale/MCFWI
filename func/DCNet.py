import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from loss import FocalFrequencyLoss


NORM_LAYERS = { 'bn': nn.BatchNorm2d, 'in': nn.InstanceNorm2d, 'ln': nn.LayerNorm }

class Conv2d(nn.Module):
    def __init__(self, dc, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(Conv2d, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.dc = dc

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):

        return self.dc(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

def createConvFunc(op_type):
    assert op_type in ['cv', 'od', 'cd'], 'unknown op type: %s' % str(op_type)
    if op_type == 'cv':
        return F.conv2d

    if op_type == 'cd':
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for cd_conv should be in 1 or 2'
            assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for cd_conv should be 3x3'
            assert padding == dilation, 'padding for cd_conv set wrong'

            weights_c = weights.sum(dim=[2, 3], keepdim=True)
            yc = F.conv2d(x, weights_c, stride=stride, padding=0, groups=groups)
            y = F.conv2d(x, weights, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y - yc
        return func
    elif op_type == 'od':
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for od_conv should be in 1 or 2'
            assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for od_conv should be 3x3'
            assert padding == dilation, 'padding for od_conv set wrong'
            shape = weights.shape
            weights = weights.view(shape[0], shape[1], -1)
            new_weights = torch.zeros_like(weights)
            new_weights[:, :, 0] = weights[:, :, 0]
            new_weights[:, :, 1] = weights[:, :, 1]
            new_weights[:, :, 2] = weights[:, :, 2]
            new_weights[:, :, 3] = weights[:, :, 3]
            new_weights[:, :, 4] = weights[:, :, 4] - weights[:, :, 0]
            new_weights[:, :, 5] = weights[:, :, 5] - weights[:, :, 1] - weights[:, :, 2]
            new_weights[:, :, 6] = weights[:, :, 6]
            new_weights[:, :, 7] = weights[:, :, 7] - weights[:, :, 6]
            new_weights[:, :, 8] = -weights[:, :, 7] - weights[:, :, 6] - weights[:, :, 5]
            weights_conv = new_weights.view(shape)

            y = F.conv2d(x, weights_conv, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y

        return func
    else:
        print('impossible to be here unless you force that')
        return None


nets = {
    'baseline': {
        'layer0':  'cv',
        'layer1':  'od',
        'layer2':  'cd',
        },
    'cv-3': {
        'layer0':  'cv',
        'layer1':  'cv',
        'layer2':  'cv',
        },
    'od-3': {
        'layer0':  'od',
        'layer1':  'od',
        'layer2':  'od',
        },
    'cd-3': {
        'layer0':  'cd',
        'layer1':  'cd',
        'layer2':  'cd',
        }
    }

def config_model(model):
    model_options = list(nets.keys())
    assert model in model_options, \
        'unrecognized model, please choose from %s' % str(model_options)

    print(str(nets[model]))

    dcs = []
    for i in range(3):
        layer_name = 'layer%d' % i
        op = nets[model][layer_name]
        dcs.append(createConvFunc(op))

    return dcs

# dcmodule
class DCModule(nn.Module):
    def __init__(self, dc, inchannel, outchannel):
        super(DCModule, self).__init__()
        self.inchannel = inchannel
        self.outchannel = outchannel
        self.conv = ConvBlock(inchannel, outchannel)
        # Depthwise Convolution
        self.conv1 = Conv2d(dc, inchannel, inchannel, kernel_size=3, padding=1, groups=inchannel, bias=False)
        self.norm = nn.BatchNorm2d(inchannel)
        self.relu2 = nn.ReLU()
        # Pointwise Convolution
        self.conv2 = nn.Conv2d(inchannel, outchannel, kernel_size=1, padding=0, bias=False)

    def forward(self, x):

        y = self.conv1(x)
        y = self.norm(y)
        y = self.relu2(y)
        y = self.conv2(y)

        if self.inchannel != self.outchannel:
            x =  self.conv(x)
        # skip connection
        y = y + x
        return y

class DCNet(nn.Module):
    def __init__(self, inchannel, dcs, sample_spatial=1.0):
        super(DCNet, self).__init__()

        filters = [32, 64, 128, 256, 512]
        self.inchannel = inchannel
        block_class1 = DCModule
        block_class2 = ConvBlock

        self.block1_1 = block_class2(self.inchannel, filters[1], kernel_size=(7, 1), stride=(2, 1), padding=(3, 0))
        self.block1_2 = block_class2(filters[1], filters[1], kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.block1_3 = block_class2(filters[1], filters[1], kernel_size=(3, 1), padding=(1, 0))
        self.block1_4 = block_class2(filters[1], filters[1], kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.block1_5 = block_class2(filters[1], filters[1], kernel_size=(3, 1), padding=(1, 0))

        self.block2_1 = block_class2(filters[1], filters[2], kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.block2_2 = block_class2(filters[2], filters[2], kernel_size=(3, 1), padding=(1, 0))
        self.block2_3 = block_class2(filters[2], filters[2], stride=2)
        self.block2_4 = block_class2(filters[2], filters[1])
        self.block2_right_1 = block_class1(dcs[0], filters[2], filters[2])
        self.block2_right_2 = block_class1(dcs[1], filters[2], filters[2])
        self.block2_right_3 = block_class1(dcs[2], filters[2], filters[2])

        self.block3_1 = block_class2(filters[2], filters[3], stride=2)
        self.block3_2 = block_class2(filters[3], filters[2])
        self.block3_right_1 = block_class1(dcs[0], filters[3], filters[3])
        self.block3_right_2 = block_class1(dcs[1], filters[3], filters[3])
        self.block3_right_3 = block_class1(dcs[2], filters[3], filters[3])

        self.block4_1 = block_class2(filters[3], filters[3], stride=2)
        self.block4_2 = block_class2(filters[3], filters[2])
        self.block4_right_1 = block_class1(dcs[0], filters[3], filters[3])
        self.block4_right_2 = block_class1(dcs[1], filters[3], filters[3])
        self.block4_right_3 = block_class1(dcs[2], filters[3], filters[3])

        self.diBlock0 = block_class2(filters[4], filters[3])
        self.diBlock1 = block_class2(filters[3], filters[2])
        self.diBlock2 = block_class2(filters[2], filters[1])
        self.lastBlock = block_class2(filters[3], filters[4], kernel_size=(8, math.ceil(70 * sample_spatial / 8)), padding=0)

        # decoder
        self.deconv1_1 = DeconvBlock(filters[4], filters[4], kernel_size=5)
        self.deconv1_2 = ConvBlock_norm(filters[4], filters[4])
        self.deconv2_1 = DeconvBlock(filters[4], filters[3], kernel_size=4, stride=2, padding=1)
        self.deconv2_2 = ConvBlock_norm(filters[3], filters[3])
        self.deconv3_1 = DeconvBlock(filters[3], filters[2], kernel_size=4, stride=2, padding=1)
        self.deconv3_2 = ConvBlock_norm(filters[2], filters[2])
        self.deconv4_1 = DeconvBlock(filters[2], filters[1], kernel_size=4, stride=2, padding=1)
        self.deconv4_2 = ConvBlock_norm(filters[1], filters[1])
        self.deconv5_1 = DeconvBlock(filters[1], filters[0], kernel_size=4, stride=2, padding=1)
        self.deconv5_2 = ConvBlock_norm(filters[0], filters[0])
        self.deconv6 = ConvBlock_Tanh(filters[0], 1)
        print('initialization done')

    def forward(self, x):
        x1 = self.block1_1(x)
        x1 = self.block1_2(x1)
        x1 = self.block1_3(x1)
        x1 = self.block1_4(x1)
        x1 = self.block1_5(x1)

        x2 = self.block2_1(x1)
        x2 = self.block2_2(x2)
        x2 = self.block2_3(x2)
        x2_0 = self.block2_4(x2)
        x2 = self.block2_right_1(x2)
        x2 = self.block2_right_2(x2)
        x2 = self.block2_right_3(x2)

        x3 = self.block3_1(x2)
        x3_0 = self.block3_2(x3)
        x3 = self.block3_right_1(x3)
        x3 = self.block3_right_2(x3)
        x3 = self.block3_right_3(x3)

        x4 = self.block4_1(x3)
        x4 = self.block4_right_1(x4)
        x4 = self.block4_right_2(x4)
        x4 = self.block4_right_3(x4)

        x = self.lastBlock(x4)

        # Decoder Part
        x = self.deconv1_1(x)
        x = self.deconv1_2(x)

        x = self.deconv2_1(x)
        offset1 = (x.size()[2] - x4.size()[2])
        offset2 = (x.size()[3] - x4.size()[3])
        padding = [offset2 // 2, (offset2 + 1) // 2, offset1 // 2, (offset1 + 1) // 2]
        outputs1 = F.pad(x4, padding)
        x = torch.cat([outputs1, x], 1)
        x = self.diBlock0(x)
        x = self.deconv2_2(x)

        x = self.deconv3_1(x)
        offset1 = (x.size()[2] - x3_0.size()[2])
        offset2 = (x.size()[3] - x3_0.size()[3])
        padding = [offset2 // 2, (offset2 + 1) // 2, offset1 // 2, (offset1 + 1) // 2]
        outputs1 = F.pad(x3_0, padding)
        x = torch.cat([outputs1, x], 1)
        x = self.diBlock1(x)
        x = self.deconv3_2(x)

        x = self.deconv4_1(x)
        offset1 = (x.size()[2] - x2_0.size()[2])
        offset2 = (x.size()[3] - x2_0.size()[3])
        padding = [offset2 // 2, (offset2 + 1) // 2, offset1 // 2, (offset1 + 1) // 2]
        outputs1 = F.pad(x2_0, padding)
        x = torch.cat([outputs1, x], 1)
        x = self.diBlock2(x)
        x = self.deconv4_2(x)

        x = self.deconv5_1(x)
        x = self.deconv5_2(x)
        x = F.pad(x, [-5, -5, -5, -5], mode="constant", value=0)
        x = self.deconv6(x)

        return x

class ConvBlock(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1, norm='bn', relu_slop=0.2, dropout=None):
        super(ConvBlock,self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.LeakyReLU(relu_slop, inplace=True))
        if dropout:
            layers.append(nn.Dropout2d(0.8))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class ConvBlock_norm(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1, norm='bn', relu_slop=0.2, dropout=None):
        super(ConvBlock_norm,self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.LeakyReLU(relu_slop, inplace=True))
        if dropout:
            layers.append(nn.Dropout2d(0.8))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class DeconvBlock(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=2, stride=2, padding=0, output_padding=0, norm='bn'):
        super(DeconvBlock, self).__init__()
        layers = [nn.ConvTranspose2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.layers = nn.Sequential(*layers)
    def forward(self, x):
        return self.layers(x)

class ConvBlock_Tanh(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1, norm='bn'):
        super(ConvBlock_Tanh, self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.Tanh())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class DCModel(nn.Module):
    """U-based network for self-reconstruction task"""
    def __init__(self):
        super(DCModel, self).__init__()
        self.dcs = config_model("cv-3") #a-v15  carv4  r16  a16  c16  hd16 (c16=150epoch; hd16=150epoch_newBlock=hd+cd;)
        self.net = DCNet(5, self.dcs)

    def forward(self, x, label_dsp_dim=None):
        x = self.net(x)
        return  x

class LossDCNet:
    def     __init__(self, weights = [1, 1]):
        '''
        Define the loss function of DCNet
        :param weights:         The weights of the two decoders in the calculation of the loss value.
        '''
        # mse
        self.criterion1 = nn.MSELoss()
        # focal loss
        self.focalLoss = FocalFrequencyLoss()

        self.weights = weights


    def __call__(self, outputs1, targets1):
        '''

        :param outputs1: Output of the real image
        :param targets1: Output of the predict image
        :return:
        '''
        mse = self.criterion1(outputs1, targets1)
        print('MSELoss:{:.12f}'.format(mse.item()), end='\t')

        loss = self.focalLoss(outputs1, targets1)
        print('FocalLoss:{:.12f}'.format(loss.item()))

        criterion = (self.weights[0] * mse + self.weights[1] * loss)

        return criterion


model_dict = {
    'DCNet': DCModel
}
