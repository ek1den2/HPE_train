import argparse

import torch

from lib.core.config import config, update_config
from lib.network.networks import get_pose_net
from lib.utils.run_utils import get_using_device, load_ckpt

import onnx
import onnxruntime as ort

parser = argparse.ArgumentParser(description='画像の姿勢推定')
parser.add_argument('--cfg', type=str, required=True, help='configファイルのパス')
args, rest = parser.parse_known_args()
update_config(args.cfg)

parser.add_argument('--ckpt', type=str, default='checkpoints/vgg2016/best_epoch.pth', help='チェックポイントのパス')
parser.add_argument('--device', type=str, default=None, choices=['cpu', 'cuda', 'mps'], help='使用するデバイス')
parser.add_argument('--onnx_path', type=str, default='checkpoints/pose_resnet.onnx', help='出力ONNXファイル名')
args = parser.parse_args()


pose_model = get_pose_net(config, is_train=False)
pose_model = load_ckpt(
    pose_model,
    args.ckpt,
    device=get_using_device(args.device)
)
pose_model.eval()

dummy_input = torch.randn(1, 3, config.MODEL.IMAGE_SIZE[1], config.MODEL.IMAGE_SIZE[0]).to(args.device)
torch.onnx.export(
    pose_model,
    dummy_input,
    args.onnx_path,
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output']
)

print(f"ONNX model exported to {args.onnx_path}")

onnx_model = onnx.load(args.onnx_path)
onnx.checker.check_model(onnx_model)

ort_sess = ort.InferenceSession(args.onnx_path)
outputs = ort_sess.run(None, {"input": dummy_input.numpy()})
print("ONNX output shape:", outputs[0].shape)