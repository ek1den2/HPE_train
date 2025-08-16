import math
import numpy as np
import cv2

import torch

from enum import Enum
from collections import OrderedDict

from lib.utils.transforms import transform_preds


class CustomPart(Enum):
    """カスタムキーポイントの定義"""
    Head = 0
    LShoulder = 1
    LElbow = 2
    LWrist = 3
    RShoulder = 4
    RElbow = 5
    RWrist = 6
    LHip = 7
    LKnee = 8
    LAnkle = 9
    RHip = 10
    RKnee = 11
    RAnkle = 12

CustomPairs = [
    (0, 1), (0, 4), (1, 4), (1, 2), (4, 5), (2, 3), (5, 6),
    (1, 7), (4, 10), (7, 10), (7, 8), (10, 11), (8, 9), (11, 12)
]

CustomColors = [
    [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], 
    [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],
    [0, 170, 255], [0, 85, 255], [0, 0, 255], [255, 0, 255]
]

CustomPairsRender = CustomPairs


def get_using_device(device=None):
    if device == 'cuda' or (device is None and torch.cuda.is_available()):
        print(">>>> Using CUDA (Nvidia) <<<<")
        return torch.device("cuda")

    elif device == 'mps' or (device is None and getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()):
        print(">>>> Using MPS (Apple Silicon GPU) <<<<")
        return torch.device("mps")

    else:
        print(">>>> Using CPU <<<<")
        return torch.device("cpu")


def extract_state_dict(obj):
    """tuple や dict を再帰的に掘って state_dict を探す"""
    if isinstance(obj, dict):
        # 'state_dict' や 'model' というキーがあればそれを返す
        for key in ['state_dict', 'model']:
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
        return obj  # dict そのものが state_dict の場合
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            result = extract_state_dict(item)
            if result is not None:
                return result
    return None  # 見つからなかった場合

def load_ckpt(model, ckpt_path, device='cuda'):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    state_dict = extract_state_dict(ckpt)
    if state_dict is None or not isinstance(state_dict, dict):
        raise TypeError(f"Could not find state_dict in checkpoint: {type(ckpt)}")

    # 'module.' を削除（マルチGPU対応）
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict, strict=True)
    model.eval()
    model.float()
    model.to(device)

    return model


def get_max_preds(outputs):
    '''
    スコアマップから予測を取得
    outputs: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    heatmap = outputs.detach().cpu().numpy()[0]

    num_joints, height, width = heatmap.shape

    # (num_joints, height * width)
    heatmap_reshaped = heatmap.reshape(num_joints, -1)

    idx = np.argmax(heatmap_reshaped, axis=1)
    maxvals = np.amax(heatmap_reshaped, axis=1)

    preds_x = idx % width
    preds_y = idx // width

    preds = np.stack([preds_x, preds_y], axis=1).astype(np.float32)
    pred_mask = (maxvals > 0.0).astype(np.float32)[:, np.newaxis]

    preds *= pred_mask

    return preds, maxvals

def get_final_preds(heatmap, center, scale):
    coords, maxvals = get_max_preds(heatmap)

    # shape 補正：バッチ次元が消えていたら復元
    if coords.ndim == 2:
        coords = coords[np.newaxis, ...]
    if heatmap.ndim == 3:
        heatmap = heatmap.unsqueeze(0)  # torch用

    heatmap_height = heatmap.shape[2]
    heatmap_width = heatmap.shape[3]
    print(f"Heatmap size: {heatmap_width}x{heatmap_height}")

    preds = coords.copy()

    for n in range(coords.shape[0]):  # batch_size
        for p in range(coords.shape[1]):  # num_joints
            hm = heatmap[n][p].cpu().numpy()
            px = int(math.floor(coords[n][p][0] + 0.5))
            py = int(math.floor(coords[n][p][1] + 0.5))

            if 1 < px < heatmap_width - 1 and 1 < py < heatmap_height - 1:
                diff = np.array([
                    hm[py][px + 1] - hm[py][px - 1],
                    hm[py + 1][px] - hm[py - 1][px]
                ])
                preds[n][p] += np.sign(diff) * 0.25
    
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(coords[i], center[i], scale[i],
                                   [heatmap_width, heatmap_height])

    return preds, maxvals



def draw_humans(npimg, outputs, maxvals, imgcopy=False):
    ''' 
    outputs: ([x0, y0], ..., [x13, y13])
    '''
    if imgcopy:
        npimg = np.copy(npimg)
    
    image_h, image_w = npimg.shape[:2]
    print(f"Image size: {image_h}x{image_w}")
    scale = (image_h + image_w) / 2.0 / 1000 # 点・線のスケール制御
    centers = {}

    for i in range(13):
        if maxvals[i] <= 0.5:
            continue

        body_part = outputs[0][i]
        center = (int(body_part[0]), int(body_part[1]))
        print(f"Part {i}: {center}, Maxval: {maxvals[i]}")
        centers[i] = center
        cv2.circle(npimg, center, 1, CustomColors[i], thickness=max(1, int(10*scale)), lineType=8)

    # 線を描画（最後2つは描画しない）
    for pair_order, pair in enumerate(CustomPairsRender):
        if maxvals[pair[0]] <= 0.5 or maxvals[pair[1]] <= 0.5:
            continue
        cv2.line(npimg, centers[pair[0]], centers[pair[1]], CustomColors[pair_order], max(1, int(2*scale)))

    return npimg

image_height = 192  # 画像の高さ
image_width = 256   # 画像の幅
aspect_ratio = image_width * 1.0 / image_height
pixel_std = 300.0  # ピクセルの標準偏差

# 2, 2, 10, 10
def xywh2cs(x, y, w, h):
    # 中心座標の計算
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5
    print(f'Center: {center}')


    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array(
        [w * 1.0 / pixel_std, h * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale