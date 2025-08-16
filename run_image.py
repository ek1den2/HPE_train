import argparse
import cv2

import numpy as np
import matplotlib.pyplot as plt

from time import time
import torch
from torchvision import transforms

from lib.core.config import config, update_config
from lib.network.networks import get_pose_net
from lib.utils.run_utils import get_using_device, load_ckpt, get_final_preds, draw_humans, get_max_preds, xywh2cs

def parse_args():
    parser = argparse.ArgumentParser(description='画像の姿勢推定')
    parser.add_argument('--cfg', type=str, required=True, help='configファイルのパス')
    args, rest = parser.parse_known_args()
    update_config(args.cfg)

    parser.add_argument('--ckpt', type=str, default='./checkpoints/vgg2016/best_epoch.pth', help='チェックポイントのパス')
    parser.add_argument('--image', type=str, default=None, help='入力画像名')
    parser.add_argument('--device', type=str, default=None, choices=['cpu', 'cuda', 'mps'], help='使用するデバイス')
    args = parser.parse_args()

    return args

def process_image(model, input_img):
    
    # TODO: 画像から人物領域のbboxを取得（YOLOなど）
    
    with torch.no_grad():
        output = model(input_img)

        plt.subplot(1, 2, 1)
        heatmaps = output[0].cpu().numpy()
        plt.imshow(heatmaps[0], cmap='jet')

        plt.subplot(1, 2, 2)
        img_norm = input_img[0].cpu()
        img_norm = img_norm.permute(1, 2, 0).numpy()
        plt.imshow(img_norm)
        plt.show()

        preds, maxvals = get_max_preds(output)
        print(preds)
        preds = scale_keypoints_around_center(preds, (48, 64), (800, 1200))
        


    return [preds], maxvals


def scale_keypoints_around_center(keypoints: list[tuple[float, float]],
                                  old_dims: tuple[int, int],
                                  new_dims: tuple[int, int]) -> np.ndarray:

    # NumPy配列に変換して、ベクトル化された計算を可能にする
    keypoints_arr = np.array(keypoints, dtype=np.float32)

    # 元のサイズと新しいサイズを取得
    w_old, h_old = old_dims
    w_new, h_new = new_dims

    # 中心座標を計算
    center_old = np.array([w_old / 2.0, h_old / 2.0])
    center_new = np.array([w_new / 2.0, h_new / 2.0])

    # 拡大・縮小率を計算
    scale_x = w_new / w_old
    scale_y = h_new / h_old
    scale_factors = np.array([scale_x, scale_y])

    translated_kps = keypoints_arr - center_old
    
    scaled_kps = translated_kps * scale_factors
    
    new_keypoints = scaled_kps + center_new

    return new_keypoints


def preprocess(np_img, args):
    # np_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
    res_img = cv2.resize(np_img, (192, 256), interpolation=cv2.INTER_AREA)
    res_img = res_img.astype('float32') / 255.0
    res_img = (res_img - 0.5) / 0.5
    res_img = torch.from_numpy(res_img).permute(2, 0, 1)

    return res_img.unsqueeze(0).to(args.device)



def main():
    args = parse_args()

    pose_model = get_pose_net(config, is_train=False)
    pose_model = load_ckpt(
        pose_model,
        args.ckpt,
        device=get_using_device(args.device)
    )

    ori_img = cv2.imread(args.image)
    

    # 前処理実行
    input_tensor = preprocess(ori_img, args)  # shape: (3, H, W)


    preds, maxvals = process_image(pose_model, input_tensor)
    out = draw_humans(ori_img, preds, maxvals)
    cv2.imwrite('output.jpg', out)


if __name__ == '__main__':
    main()