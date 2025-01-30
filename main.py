import os
import torch
from datasets import DTERDataset
import log
import argparse
from checkpoint import load_checkpoint
from torch.utils.data import DataLoader
from ml_opts import train, evaluate
from models import get_detr


args = argparse.ArgumentParser()
args.add_argument("--experiment_name", help="name_of_experiment")
args.add_argument("--experiments_path", default="experiments", help="path to save experiments")

args.add_argument("--train_path", help="path for train directory")
args.add_argument("--test_path", help="path for test directory")

args.add_argument("--bbox_format", default="cxcywh", type=str, help="dataset's bbox format, default is standard yolo format")
args.add_argument("--normalized", default=True, help="if true means that the bboxes are already normalized, "
                                                     "else the dataset will normalize them")

args.add_argument("--train_batch_size", default=32, type=int, help="size of batch for train")
args.add_argument("--test_batch_size", default=32, type=int, help="size of batch for test")
args.add_argument("--device", default="cuda", type=str, help="device for training and testing")
args.add_argument("--lr", default=1e-5, type=float, help="learning rate")
args.add_argument("--weight_decay", default=1e-4, type=float, help="weight decay")
args.add_argument("--enable_amp", default=False, type=bool, help="if true use automatic mixed precision")
args.add_argument("--resume_path", default=None, type=str, help="path to a checkpoint for resume training")
args.add_argument("--epochs", default=20, type=int, help="number of epochs")
args.add_argument("--save_every", default=2, type=int, help="save checkpoint every number of epochs")
args.add_argument("--eval_mode", default=False, action="store_true", help="if true, only eval model. "
                                                                          "Note that resume path is needed else "
                                                                          "it will evaluate an empty model")
args.add_argument("--num_classes", default=3, type=int, help="number of classes in the dataset")
# args.add_argument("--num_workers", default=4, type=int, help="number of workers")


args = args.parse_args()

os.makedirs(f"{args.experiments_path}/{args.experiment_name}", exist_ok=True)
log_file = f"{args.experiments_path}/{args.experiment_name}/log_file.txt"
log.creat_logger(log_file)

args.num_classes += 1


if __name__ == '__main__':
    model, processor = get_detr(args.num_classes, args.device)

    # Dataset and DataLoader
    train_dataset = DTERDataset(
        image_dir=f"{args.train_path}/images",
        label_dir=f"{args.train_path}/labels",
        processor=processor,
        bbox_format=args.bbox_format,
        normalized=args.normalized
    )

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True,
                              collate_fn=lambda x: tuple(zip(*x)))

    test_dataset = DTERDataset(
        image_dir=f"{args.test_path}/images",
        label_dir=f"{args.test_path}/labels",
        processor=processor,
        bbox_format=args.bbox_format,
        normalized=args.normalized
    )

    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False,
                             collate_fn=lambda x: tuple(zip(*x)))

    scaler = torch.cuda.amp.GradScaler(enabled=args.enable_amp)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)  # 1e-4)

    # Main Training Process
    start_epoch = 0
    if args.resume_path is not None:
        start_epoch = load_checkpoint(args.resume_path, model, optimizer, scaler)

    if args.eval_mode:
        evaluate(model, test_loader, args.device)
    else:
        train(model, train_loader, test_loader, optimizer, scaler, start_epoch, args)
