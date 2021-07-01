import os
import glob
import argparse

import tqdm
import torch
import torch.nn as nn
import wandb
import numpy as np

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from model.cbdnet import Network, fixed_loss
from utils.common import AverageMeter, psnr_score
from dataset.loader import BaseDataset


def parse_args():

    parser = argparse.ArgumentParser(description="Dacon LG Contest")
    parser.add_argument(
        "-a", "--arch", default="GMSNet", type=str, help="model architecture"
    )
    parser.add_argument("-b", "--batch-size", default=16, type=int, help="batch size")
    parser.add_argument("-j", "--job", default=4, type=int, help="number of workers")
    parser.add_argument("-e", "--epochs", default=6000, type=int, help="max epoch")
    parser.add_argument("-s", "--step", default=1000, type=int, help="step size")
    parser.add_argument(
        "--learning-rate", default=2e-4, type=float, help="learning rate"
    )

    args = parser.parse_args()

    return args


def train_batch(batch, model, criterion, optimizer, device):

    model.train()

    images = batch[0].to(device)
    labels = batch[1].to(device)

    noise_level_est, outputs = model(images)
    loss = criterion(outputs, labels, noise_level_est, 0, 0)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.data.cpu().item(), psnr_score(
        outputs.data.cpu().detach().numpy(), labels.cpu().detach().numpy()
    )


def valid_batch(valid_loader, model, criterion, device):

    model.eval()

    valid_loss = AverageMeter()
    valid_psnr_scores = AverageMeter()

    for batch in valid_loader:

        image = batch[0].to(device)
        label = batch[1].to(device)

        noise_level_est, outputs = model(image)
        loss = criterion(outputs, label, noise_level_est, 0, 0)

        valid_loss.update(loss.data.cpu().item(), len(image))
        valid_psnr_scores.update(
            psnr_score(
                outputs.data.cpu().detach().numpy(), label.cpu().detach().numpy()
            ),
            len(image),
        )

    val_loss = valid_loss.avg
    val_psnr_score = valid_psnr_scores.avg

    return val_loss, val_psnr_score


def train():

    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    wandb.login()
    wandb.init(project="Dacon", entity="coco1578", name=args.arch)

    global_step = 0
    best_loss = np.inf

    model = Network()
    model_info = torch.load(
        "/home/salmon21/LG/WeCanBeLGMan/save_model/cbdnet/checkpoint.pth.tar",
        map_location="cpu",
    )
    model.load_state_dict(model_info["state_dict"], strict=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    criterion = fixed_loss()

    model, criterion = model.to(device), criterion.to(device)

    inputs = sorted(glob.glob("/home/salmon21/LG/dataset/train/input/204_102/*.npy"))
    labels = sorted(glob.glob("/home/salmon21/LG/dataset/train/label/204_102/*.npy"))

    X_train, X_test, y_train, y_test = train_test_split(
        inputs, labels, test_size=0.25, random_state=5252
    )

    train_dataset = BaseDataset(X_train, y_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=args.job,
    )
    valid_dataset = BaseDataset(X_test, y_test, train=False)
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.job,
    )

    train_losses = AverageMeter()
    train_psnr_scores = AverageMeter()

    for epoch in tqdm.tqdm(range(args.epochs)):
        progress_bar = tqdm.tqdm(
            enumerate(train_loader),
            total=int(len(train_loader) / args.batch_size),
            position=0,
            leave=True,
        )

        for index, batch in progress_bar:

            train_loss, train_psnr_score = train_batch(
                batch, model, criterion, optimizer, device
            )
            train_losses.update(train_loss, len(batch[0]))
            train_psnr_scores.update(train_psnr_score, len(batch[0]))
            global_step += 1

            description = f"EPOCH:[{epoch + 1}] - STEP:[{global_step}] - LOSS:[{train_losses.avg:.4f}]"
            progress_bar.set_description(description)

            if global_step % args.step == 0:

                with torch.no_grad():
                    valid_loss, valid_psnr_score = valid_batch(
                        valid_loader, model, criterion, device
                    )

                    if valid_loss < best_loss:

                        best_loss = valid_loss

                        torch.save(
                            {
                                "loss": best_loss,
                                "psnr_score": valid_psnr_score,
                                "state_dict": model.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "scheduler": scheduler.state_dict(),
                            },
                            os.path.join(
                                "/home/salmon21/LG/WeCanBeLGMan/save_model/gmsnet",
                                "checkpoint.pth",
                            ),
                        )
                scheduler.step()

                wandb.log(
                    {
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/loss": train_losses.avg,
                        "train/psnr_score": train_psnr_score,
                        "valid/loss": valid_loss,
                        "valid/psnr_score": valid_psnr_score,
                        "global_steps": global_step,
                    }
                )

                train_losses.reset()
                train_psnr_scores.reset()

            else:
                wandb.log({"global_steps": global_step})


if __name__ == "__main__":

    train()
