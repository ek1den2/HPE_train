from lib.network import pose_resnet
from lib.network import pose_mobilenet

def get_pose_net(cfg, is_train):
    if cfg.MODEL.NAME == 'pose_resnet':
        return pose_resnet.get_model(cfg, is_train)
    elif cfg.MODEL.NAME == 'pose_mobilenet':
        return pose_mobilenet.get_model(cfg, is_train)
    else:
        raise ValueError(f"モデル名がサポートされていません: {cfg.MODEL.NAME}")