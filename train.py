
import argparse
import os
import pprint
from time import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from torch.utils.tensorboard import SummaryWriter

from lib.core.config import config, get_model_name, update_config
from lib.core.loss import JointsMSELoss
from lib.utils.utils import get_optimizer, save_checkpoint, data_loader
from lib.utils.utils import create_logger, suppress_all_output
from lib.core.function import train, validate

from lib.network.networks import get_pose_net


def parse_args():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--cfg', type=str, required=True, help='yamlファイルのパス')
    args, rest = parser.parse_known_args()
    update_config(args.cfg)

    # training
    parser.add_argument('--freq', type=int, default=config.PRINT_FREQ, help='logの頻度')
    parser.add_argument('--gpus', type=str, help='gpus')
    parser.add_argument('--workers', type=int, help='データローダーのワーカー数')
    args = parser.parse_args()

    return args

def reset_config(config, args):
    if args.freq:
        config.PRINT_FREQ = args.freq
    if args.gpus:
        config.GPUS = args.gpus
    if args.workers:
        config.WORKERS = args.workers


def main():
    args = parse_args()
    reset_config(config, args)
    
    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'train'
    )

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn設定
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    model = get_pose_net(config, is_train=True)

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    dump_input = torch.rand((
        config.TRAIN.BATCH_SIZE,
        3,
        config.MODEL.IMAGE_SIZE[0],
        config.MODEL.IMAGE_SIZE[1]
        ))
    
    writer_dict['writer'].add_graph(model, (dump_input, ), verbose=False)

    gpus = [int(i) for i in config.GPUS.split(',')]
    model = nn.DataParallel(model, device_ids=gpus).cuda()

    # 損失関数・最適化関数を定義
    criterion = JointsMSELoss(
        use_target_weight=config.LOSS.USE_TARGET_WEIGHT
    ).cuda()

    optimizer = get_optimizer(config, model)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR
    )

    train_dataset, train_loader, valid_dataset, valid_loader = data_loader(config, gpus)

    best_perf = 0.0
    best_model = False

    # 学習開始
    print("\nvvvvvvvvvvv Start Training vvvvvvvvvvv\n")
    for epoch in range(config.TRAIN.BEGIN_EPOCH, config.TRAIN.END_EPOCH):
        start = time()
        train_loss, train_acc = train(config, train_loader, model, criterion, optimizer, epoch,
                          final_output_dir, tb_log_dir, writer_dict)
        
        with suppress_all_output():
            pref_indicator, val_loss, val_acc = validate(config, valid_loader, valid_dataset, model,
                                    criterion, final_output_dir, tb_log_dir, writer_dict)

        lr = optimizer.param_groups[0]['lr']
        elapsed_time = time() - start
        print(f'[{epoch+1}] time {elapsed_time:.2f} lr {lr:.6g} [train] loss:{train_loss:.6f} acc: {train_acc:.6f} [val] loss:{val_loss:.6f} acc: {val_acc:.6f} mAP: {pref_indicator:.6f}')

        lr_scheduler.step()
        
        is_best = pref_indicator > best_perf
        if is_best:
            best_perf = pref_indicator
            save_checkpoint({
                'epoch': epoch + 1,
                'model': get_model_name(config),
                'state_dict': model.state_dict(),
                'pref': pref_indicator,
                'optimizer': optimizer.state_dict(),
            }, best_model, final_output_dir, filename=f'best_epoch.pth')
            print('=> saving best checkpoint')

        final_model_state_file = os.path.join(final_output_dir,
                                            'final_state.pth')
        # logger.info('=> saving final checkpoint')
        torch.save(model.module.state_dict(), final_model_state_file)
        writer_dict['writer'].close()

    print("\n^^^^^^^^^^^ End Training ^^^^^^^^^^^^\n")

if __name__ == '__main__':
    main()
