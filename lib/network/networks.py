from lib.network import pose_resnet
from lib.network import pose_mobilenet

def get_pose_net(cfg, is_train):
    if cfg.MODEL.NAME == 'pose_resnet':
        return pose_resnet.get_model(cfg, is_train)
    elif cfg.MODEL.NAME == 'pose_mobilenet':
        return pose_mobilenet.get_model(cfg, is_train)
    else:
        raise ValueError(f"モデル名がサポートされていません: {cfg.MODEL.NAME}")
    



if __name__ == '__main__':
    import torch
    import thop
    from train import parse_args, reset_config
    from lib.core.config import config
    from torch.utils.tensorboard import SummaryWriter
    args = parse_args()
    reset_config(config, args)

    model = get_pose_net(config, is_train=True)

    print(model)

    dummy_input = torch.randn(1, 1, 160, 160)
    flops, params = thop.profile(model, inputs=(dummy_input, ))
    gflops = flops / 1e9
    print(f"FLOPs: {flops} ({gflops:.2f} GFLOPs)")
    print(f"Parameters: {params}")

    writer = SummaryWriter("experiments/tbX/")
    writer.add_graph(model, (dummy_input, ))
    writer.close()