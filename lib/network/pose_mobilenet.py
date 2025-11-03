import os
import logging

import torch
import torch.nn as nn
from collections import OrderedDict

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

# conv + bn + relu
class ConvBN(nn.Module):
    def __init__(self, nin, nout, kernel_size=3, stride=1, padding=1, bias=False, relu=True):
        super(ConvBN, self).__init__()
        self.conv = nn.Conv2d(nin, nout, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(nout)
        self.activation = nn.ReLU(inplace=True) if relu else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

# Depthwise Separable Convolution（depthwise + bn + pointwise + bn）
class DSConv(nn.Module):
    def __init__(self, nin, nout, kernel_size=3, stride=1, padding=1, bias=False, relu=True):
        super(DSConv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, stride=stride, padding=padding, groups=nin, bias=bias)
        self.bn1 = nn.BatchNorm2d(nin)
        self.relu1 = nn.ReLU(inplace=True)

        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(nout)
        self.relu2 = nn.ReLU(inplace=True) if relu else nn.Identity()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


class MobileNet(nn.Module):
    """MobileNetV1 モデル"""
    
    def __init__(self, in_channels=64, conv_width=1.0):
        super(MobileNet, self).__init__()
        print("Building MobileNet")

        self.conv_width = conv_width
        min_depth = 8
        depth = lambda d: max(round(d * self.conv_width), min_depth)

        # MobileNetバックボーン（前処理ステージ）
        self.model0 = nn.Sequential(
            DSConv(depth(in_channels), depth(64), 3, 1, 1),       # index 1
            DSConv(depth(64), depth(128), 3, 2, 1),      # index 2
            DSConv(depth(128), depth(128), 3, 1, 1),     # index 3
            DSConv(depth(128), depth(256), 3, 2, 1),     # index 4
            DSConv(depth(256), depth(256), 3, 1, 1),     # index 5
            DSConv(depth(256), depth(512), 3, 2, 1),     # index 6
            DSConv(depth(512), depth(512), 3, 1, 1),     # index 7
            DSConv(depth(512), depth(512), 3, 1, 1),     # index 8
            DSConv(depth(512), depth(512), 3, 1, 1),     # index 9
            DSConv(depth(512), depth(512), 3, 1, 1),     # index 10
            DSConv(depth(512), depth(512), 3, 1, 1)      # index 11
        )

    def forward(self, x):
        return self.model0(x)

class PoseMobileNet(nn.Module):
    def __init__(self, cfg):
        extra = cfg.MODEL.EXTRA
        self.deconv_with_bias = extra.DECONV_WITH_BIAS
        self.inplanes = 512

        super(PoseMobileNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.mobile_net = MobileNet(in_channels=64, conv_width=1.0)

        # deconv
        self.deconv_layers = self._make_deconv_layer(
            extra.NUM_DECONV_LAYERS,
            extra.NUM_DECONV_FILTERS,
            extra.NUM_DECONV_KERNELS,
        )

        self.final_layer = nn.Conv2d(
            in_channels=extra.NUM_DECONV_FILTERS[-1],
            out_channels=cfg.MODEL.NUM_JOINTS,
            kernel_size=extra.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
        )
    
    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        # MobileNet部分
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.mobile_net(x)

        # Deconv
        x = self.deconv_layers(x)
        x = self.final_layer(x)

        return x
    

    def init_weights(self, pretrained=''):
        if os.path.isfile(pretrained):
            # 事前学習済みモデルの読み込み
            logger.info(f'=> loading pretrained model {pretrained}')
            checkpoint = torch.load(pretrained)

            if isinstance(checkpoint, OrderedDict):
                state_dict = checkpoint
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            
            missing_keys, unexpected_keys = self.load_state_dict(new_state_dict, strict=False)
            logger.info(f'=> missing keys: {missing_keys}')
            logger.info(f'=> unexpected keys: {unexpected_keys}')
        
        else:
            # スクラッチからの初期化
            logger.info('=> initializing weights from scratch')
            self._init_encoder_weights()
            self._init_decoder_weights()


    def _init_encoder_weights(self):
        """ エンコーダの重みを初期化 """
        logger.info('=> initializing encoder weights')
        for m in [self.conv1, self.bn1, self.mobile_net]:
            for mod in m.modules():
                if isinstance(mod, nn.Conv2d):
                    nn.init.kaiming_normal_(mod.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(mod, nn.BatchNorm2d):
                    nn.init.constant_(mod.weight, 1)
                    nn.init.constant_(mod.bias, 0)

    def _init_decoder_weights(self):
        """ デコーダの重みを初期化 """
        logger.info('=> initializing decoder weights')
        for m in self.deconv_layers.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                if self.deconv_with_bias:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        logger.info('=> initializing final layer with Gaussian')
        for m in self.final_layer.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)


def get_model(cfg, is_train):

    model = PoseMobileNet(cfg)

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED)
    
    return model