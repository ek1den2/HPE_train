import os
import logging

import torch
import torch.nn as nn
from collections import OrderedDict

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

# conv + bn + relu6
class ConvBN(nn.Module):
    def __init__(self, nin, nout, kernel_size=3, stride=1, padding=1, bias=False, relu=True):
        super(ConvBN, self).__init__()
        self.conv = nn.Conv2d(nin, nout, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(nout)
        self.relu6 = nn.ReLU6(inplace=True) if relu else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu6(x)
        return x

# 1x1 conv
class Conv1x1BN(nn.Module):
    def __init__(self, nin, nout, bias=False, relu=True):
        super(Conv1x1BN, self).__init__()
        self.conv = nn.Conv2d(nin, nout, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bn = nn.BatchNorm2d(nout)
        self.relu6 = nn.ReLU6(inplace=True) if relu else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu6(x)
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

class IRB(nn.Module):
    def __init__(self, nin, nout, stride, expand_ratio):
        super(IRB, self).__init__()
        self.stride = stride
        assert stride in [1, 2] # ストライドは1 or 2

        hidden_dim = int(round(nin * expand_ratio))
        self.use_res_connect = self.stride == 1 and nin == nout

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, nout, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(nout),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(nin, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, nout, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(nout),
            )
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    """MobileNetV2 モデル"""
    
    def __init__(self, conv_width=1.0):
        super(MobileNetV2, self).__init__()
        print("Building MobileNetV2")

        self.conv_width = conv_width
        min_depth = 8
        depth = lambda d: max(round(d * self.conv_width), min_depth)

        # 1層目
        self.features = ConvBN(1, depth(32), stride=2, padding=1, bias=False)

        # 2-18層目
        self.irblock1 = IRB(depth(32), depth(16), stride=1, expand_ratio=1)    # 1  n = 1
        self.irblock2 = IRB(depth(16), depth(24), stride=2, expand_ratio=6)    # 2  n = 2
        self.irblock3 = IRB(depth(24), depth(24), stride=1, expand_ratio=6)    # 3
        self.irblock4 = IRB(depth(24), depth(32), stride=2, expand_ratio=6)    # 4  n = 3
        self.irblock5 = IRB(depth(32), depth(32), stride=1, expand_ratio=6)    # 5
        self.irblock6 = IRB(depth(32), depth(32), stride=1, expand_ratio=6)    # 6
        self.irblock7 = IRB(depth(32), depth(64), stride=2, expand_ratio=6)    # 7  n = 4
        self.irblock8 = IRB(depth(64), depth(64), stride=1, expand_ratio=6)    # 8
        self.irblock9 = IRB(depth(64), depth(64), stride=1, expand_ratio=6)    # 9
        self.irblock10 = IRB(depth(64), depth(64), stride=1, expand_ratio=6)   # 10
        self.irblock11 = IRB(depth(64), depth(96), stride=1, expand_ratio=6)   # 11  n = 3
        self.irblock12 = IRB(depth(96), depth(96), stride=1, expand_ratio=6)   # 12
        self.irblock13 = IRB(depth(96), depth(96), stride=1, expand_ratio=6)   # 13
        self.irblock14 = IRB(depth(96), depth(160), stride=2, expand_ratio=6)  # 14  n = 3
        self.irblock15 = IRB(depth(160), depth(160), stride=1, expand_ratio=6) # 15
        self.irblock16 = IRB(depth(160), depth(160), stride=1, expand_ratio=6) # 16
        self.irblock17 = IRB(depth(160), depth(320), stride=1, expand_ratio=6) # 17  n = 1

        # self.avgpool = nn.AdaptiveAvgPool2d((7, 7)) # 7x7の適応平均プーリング

        # 最終層
        self.final_layer = (Conv1x1BN(depth(320), 512, bias=False))    # 18
    
    def forward(self, x):
        out0 = self.features(x)  # 1層目
        out1 = self.irblock1(out0)  # 2層目
        out2 = self.irblock2(out1)  # 3層目
        out3 = self.irblock3(out2)  # 4層目
        out4 = self.irblock4(out3)  # 5層目
        out5 = self.irblock5(out4)  # 6層目
        out6 = self.irblock6(out5)  # 7層目
        out7 = self.irblock7(out6)  # 8層目
        out8 = self.irblock8(out7)  # 9層目
        out9 = self.irblock9(out8)  # 10層目
        out10 = self.irblock10(out9)  # 11層目
        out11 = self.irblock11(out10)  # 12層目
        out12 = self.irblock12(out11)  # 13層目
        out13 = self.irblock13(out12)  # 14層目
        out14 = self.irblock14(out13)  # 15層目
        out15 = self.irblock15(out14)  # 16層目
        out16 = self.irblock16(out15)  # 17層目
        out17 = self.irblock17(out16)  # 18層目
        outputs = self.final_layer(out17)  # 19層目
        # 7層目とアップサンプリングした14層目を結合
        # out13_upsample = nn.functional.interpolate(out13, size=out6.shape[2:], mode='bilinear', align_corners=False)
        # outputs = torch.cat([out6, out13_upsample], dim=1)

        return outputs



class PoseMobileNetV2(nn.Module):
    def __init__(self, cfg):
        extra = cfg.MODEL.EXTRA
        self.deconv_with_bias = extra.DECONV_WITH_BIAS
        self.inplanes = 512

        super(PoseMobileNetV2, self).__init__()
        self.mobile_net = MobileNetV2(conv_width=extra.CONV_WIDTH)

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
        for m in [self.mobile_net]:
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

    model = PoseMobileNetV2(cfg)

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED)
    
    return model