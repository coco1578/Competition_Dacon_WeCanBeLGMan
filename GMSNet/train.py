import os
import glob
import argparse

import tqdm
import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from model.gmsnet import GMSNet
from utils.common import AverageMeter, psnr_score
from dataset.loader import BaseDataset


def parse_args():

    parser = argparse.ArgumentParser(description="Dacon LG Contest")
    parser.add_argument("-b", "--batch-size", default=32, type=int, help="batch size")
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

    outputs = model(images)
    loss = criterion(outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.data.cpu().item()


def valid_batch(valid_loader, model, criterion, device):

    model.eval()

    valid_loss = AverageMeter()

    for batch in valid_loader:

        image = batch[0].to(device)
        label = batch[1].to(device)

        outputs = model(image)
        loss = criterion(outputs, label)

        valid_loss.update(loss.data.cpu().item(), len(image))

    val_loss = valid_loss.avg

    return val_loss


def train():

    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    global_step = 0
    best_loss = np.inf

    model = GMSNet()
    model_info = torch.load("", map_location="cpu")
    model.load_state_dict(model_info["state_dict"])
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(model_info["optimizer"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    criterion = nn.L1Loss()

    model, criterion = model.to(device), criterion.to(device)

    inputs = sorted(glob.glob("/home/salmon21/LG/dataset/train/input/*.npy"))
    labels = sorted(glob.glob("/home/salmon21/LG/dataset/train/label/*.npy"))

    X_train, X_test, y_train, y_test = train_test_split(
        inputs, labels, test_size=0.3, random_state=5252
    )

    train_dataset = BaseDataset(X_train, y_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=args.job,
    )
    valid_dataset = BaseDataset(X_test, y_test)
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.job,
    )

    train_losses = AverageMeter()

    for epoch in tqdm.tqdm(range(args.epochs)):
        progress_bar = tqdm.tqdm(
            enumerate(train_loader),
            total=int(len(train_loader) / args.batch),
            position=0,
            leave=True,
        )

        for index, batch in progress_bar:

            train_loss = train_batch(batch, model, criterion, optimizer, device)
            train_losses.update(train_loss, len(batch[0]))
            global_step += 1

            description = f"EPOCH:[{epoch + 1}] - STEP:[{global_step}] - LOSS:[{train_loss.avg:.4f}]"
            progress_bar.set_description(description)

            if global_step % args.step == 0:

                with torch.no_grad():
                    valid_loss = valid_batch(valid_loader, model, criterion, device)

                    if valid_loss < best_loss:

                        best_loss = valid_loss

                        torch.save(
                            {
                                "loss": best_loss,
                                "state_dict": model.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "scheduler": scheduler.state_dict(),
                            },
                            os.path.join("/home/salmon21/LG/", "checkpoint.pth"),
                        )
                scheduler.step()
                train_loss.reset()
