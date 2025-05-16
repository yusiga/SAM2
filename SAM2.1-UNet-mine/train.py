import os
import argparse
import random
import numpy as np
import torch
import torch.optim as opt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset import FullDataset
from SAM2UNet import SAM2UNet

parser = argparse.ArgumentParser("SAM2-UNet")
parser.add_argument("--hiera_path", type=str, required=True,
                    help="path to the sam2 pretrained hiera")
parser.add_argument("--train_image_path", type=str, required=True,
                    help="path to the image that used to train the model")
parser.add_argument("--train_mask_path", type=str, required=True,
                    help="path to the mask file for training")
parser.add_argument("--val_image_path", type=str, required=True,
                    help="path to the image used for validation")
parser.add_argument("--val_mask_path", type=str, required=True,
                    help="path to the mask file for validation")
parser.add_argument('--save_path', type=str, required=True,
                    help="path to store the checkpoint")
parser.add_argument("--epoch", type=int, default=20,
                    help="training epochs")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--batch_size", default=12, type=int)
parser.add_argument("--weight_decay", default=5e-4, type=float)
args = parser.parse_args()


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def main(args):
    # 日志文件路径
    log_path = os.path.join(args.save_path, "train_log.txt")
    os.makedirs(args.save_path, exist_ok=True)

    # 打开日志文件
    log_file = open(log_path, "w")

    # dataset = FullDataset(args.train_image_path, args.train_mask_path, 352, mode='train')
    # dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)

    train_dataset = FullDataset(args.train_image_path, args.train_mask_path, 352, mode='train')
    val_dataset = FullDataset(args.val_image_path, args.val_mask_path, 352, mode='val')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

    device = torch.device("cuda")
    model = SAM2UNet(args.hiera_path)
    model.to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    # 打印确认训练的参数
    log_file.write("Trainable parameters:\n")
    for name, param in model.named_parameters():
        if param.requires_grad:
            line = f"  {name}\n"
            log_file.write(line)
    total_params = sum(p.numel() for p in trainable_params)
    log_file.write(f"Total trainable params: {total_params}\n")

    optim = opt.AdamW([{"params": trainable_params, "initia_lr": args.lr}], lr=args.lr,
                      weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optim, args.epoch, eta_min=1.0e-7)
    os.makedirs(args.save_path, exist_ok=True)
    for epoch in range(args.epoch):
        model.train()
        for i, batch in enumerate(train_loader):
            x = batch['image']
            target = batch['label']
            x = x.to(device)
            target = target.to(device)
            optim.zero_grad()
            pred0, pred1, pred2 = model(x)
            loss0 = structure_loss(pred0, target)
            loss1 = structure_loss(pred1, target)
            loss2 = structure_loss(pred2, target)
            loss = loss0 + loss1 + loss2
            loss.backward()
            optim.step()
            if i % 50 == 0:
                log_str = "epoch:{}-{}: loss:{}".format(epoch + 1, i + 1, loss.item())
                print(log_str)
                log_file.write(log_str + "\n")

        scheduler.step()

        model.eval()
        val_loss_total = 0
        with torch.no_grad():
            for val_batch in val_loader:
                x_val = val_batch['image'].to(device)
                target_val = val_batch['label'].to(device)
                pred0, pred1, pred2 = model(x_val)
                val_loss = (structure_loss(pred0, target_val) +
                            structure_loss(pred1, target_val) +
                            structure_loss(pred2, target_val))
                val_loss_total += val_loss.item()

        avg_val_loss = val_loss_total / len(val_loader)
        log_str = f"Epoch {epoch + 1}: Validation Loss: {avg_val_loss}"
        print(log_str)
        log_file.write(log_str + "\n")


        if (epoch + 1) % 5 == 0 or (epoch + 1) == args.epoch:
            torch.save(model.state_dict(), os.path.join(args.save_path, 'SAM2-UNet-%d.pth' % (epoch + 1)))
            print('[Saving Snapshot:]', os.path.join(args.save_path, 'SAM2-UNet-%d.pth' % (epoch + 1)))
    log_file.close()


# def seed_torch(seed=1024):
# 	random.seed(seed)
# 	os.environ['PYTHONHASHSEED'] = str(seed)
# 	np.random.seed(seed)
# 	torch.manual_seed(seed)
# 	torch.cuda.manual_seed(seed)
# 	torch.cuda.manual_seed_all(seed)
# 	torch.backends.cudnn.benchmark = False
# 	torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    # seed_torch(1024)
    # print(f"可见的GPU数量: {torch.cuda.device_count()}")
    # print(f"当前使用的GPU: {torch.cuda.get_device_name(0)}")
    # print(f"CUDA_VISIBLE_DEVICES 设置为: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    main(args)
