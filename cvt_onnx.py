import argparse

import torch

from collections import OrderedDict
from lib.core.config import config, update_config
from lib.network.networks import get_pose_net

import onnx
import onnxruntime as ort

def main():
    parser = argparse.ArgumentParser(description='画像の姿勢推定')
    parser.add_argument('--cfg', type=str, required=True, help='configファイルのパス')
    args, rest = parser.parse_known_args()
    update_config(args.cfg)

    parser.add_argument('--ckpt', type=str, default='checkpoints/vgg2016/best_epoch.pth', help='チェックポイントのパス')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda', 'mps'], help='使用するデバイス')
    parser.add_argument('--onnx_path', type=str, default='onnx/irpose.onnx', help='出力ONNXファイル名')
    args = parser.parse_args()


    pose_model = get_pose_net(config, is_train=False)
    pose_model = load_ckpt(
        pose_model,
        args.ckpt,
        device=get_using_device(args.device)
    )
    pose_model.eval()

    dummy_input = torch.randn(1, 1, config.MODEL.IMAGE_SIZE[1], config.MODEL.IMAGE_SIZE[0]).to(args.device)
    torch.onnx.export(
        pose_model,
        dummy_input,
        args.onnx_path,
        export_params=True,
        opset_version=16,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output']
    )

    print(f"ONNX model exported to {args.onnx_path}")

    onnx_model = onnx.load(args.onnx_path)
    onnx.checker.check_model(onnx_model)

    ort_sess = ort.InferenceSession(args.onnx_path)
    outputs = ort_sess.run(None, {"input": dummy_input.cpu().numpy()})
    print("ONNX output shape:", outputs[0].shape)


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

def extract_state_dict(obj):
    if isinstance(obj, dict):
        for key in ['state_dict', 'model']:
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
        return obj
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            result = extract_state_dict(item)
            if result is not None:
                return result
    return None

if __name__ == '__main__':
    main()