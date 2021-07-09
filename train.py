import os
import glob
import argparse

import cv2
import tqdm
import wandb
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp

from sklearn.model_selection import train_test_split
from torch.utils.data.dataloader import DataLoader

from dataset import ImageDataset
from CBDNet.model.cbdnet import Network, fixed_loss
from utils import recover_image, psnr_score, AverageMeter


def parse_args():

    parser = argparse.ArgumentParser("Dacon LG Contest")
    parser.add_argument("--backbone", type=str, default="efficientnet-b1")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument(
        "--project", type=str, default="UnetPlusPlus", help="project name of wandb"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="effiunet",
        help="model directory name. recommand to use model name. i.e.) effiunet",
    )

    return parser.parse_args()


def train_on_batch(batch, model, criterion, optimizer, device):

    model.train()

    images = batch[0].to(device)
    labels = batch[1].to(device)

    outputs = model(images)
    loss = criterion(outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.data.cpu().item()


def valid_on_batch(
    X_test_dir, y_test_dir, model, criterion, device, transformer, epoch
):

    valid_losses = []
    valid_psnr_scores = []

    model.eval()

    # TODO: Change hard coding part
    save_image_foldre = os.path.join(
        "/home/salmon21/LG/WeCanBeLGMan/save_model/effiunet", str(epoch)
    )
    os.makedirs(save_image_foldre)

    for X_tes_dir, y_tes_dir in tqdm.tqdm(zip(X_test_dir, y_test_dir)):
        valid_loss = []
        top_lefts = np.load(os.path.join(X_tes_dir, "top_lefts", "top_lefts.npy"))
        image_number = os.path.split(y_tes_dir[:-1])[1].split("_")[0]
        result_y_image = cv2.imread(
            f"/home/salmon21/LG/dataset/train/label/train_label_{image_number}.png"
        )

        # If valid image has small image size (1224, 1632, 3)
        if top_lefts[-1][0] < 2000:
            result_X_image = np.zeros((1224, 1632, 3))
            result_X_masks = np.zeros((1224, 1632, 3))
        else:
            result_X_image = np.zeros((2448, 3264, 3))
            result_X_masks = np.zeros((2448, 3264, 3))

        for i in range(len(top_lefts)):

            X_tes = np.load(os.path.join(X_tes_dir, f"{i}.npy"))
            y_tes = np.load(os.path.join(y_tes_dir, f"{i}.npy"))

            X_tes = transformer(X_tes)
            X_tes = X_tes.unsqueeze(dim=0)
            X_tes = X_tes.to(device)

            output = model(X_tes)
            loss = criterion(output, transformer(y_tes).unsqueeze(dim=0).to(device))

            output = output.cpu().detach().squeeze().permute(1, 2, 0).numpy()
            output = output * 255.0

            h, w, c = result_X_image[
                top_lefts[i][0] : top_lefts[i][0] + 256,
                top_lefts[i][1] : top_lefts[i][1] + 256,
                :,
            ].shape

            result_X_image[
                top_lefts[i][0] : top_lefts[i][0] + 256,
                top_lefts[i][1] : top_lefts[i][1] + 256,
                :,
            ] += output[:h, :w]
            result_X_masks[
                top_lefts[i][0] : top_lefts[i][0] + 256,
                top_lefts[i][1] : top_lefts[i][1] + 256,
                :,
            ] += 1

            valid_loss.append(loss.data.cpu().item())

        result_X_image = result_X_image / result_X_masks
        result_X_image = np.uint8(result_X_image)

        valid_losses.append(np.mean(valid_loss))
        valid_psnr_scores.append(
            psnr_score(result_X_image.astype(float), result_y_image.astype(float))
        )

    return np.mean(valid_losses), np.mean(valid_psnr_scores)


def train():

    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    wandb.login()
    wandb.init(project="Dacon", entity="coco1578", name=args.project)

    model = smp.UnetPlusPlus(
        encoder_name=args.backbone,
        encoder_weights="imagenet",
        in_channels=3,
        classes=3,
        activation="sigmoid",
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": args.learning_rate * 0.1},
            {"params": model.decoder.parameters(), "lr": args.learning_rate},
        ]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=10, eta_min=1e-5
    )
    transformer = transforms.Compose([transforms.ToTensor()])

    criterion = nn.L1Loss()

    model, criterion = model.to(device), criterion.to(device)

    inputs = sorted(glob.glob("/home/salmon21/LG/dataset/train/input/**/"))
    labels = sorted(glob.glob("/home/salmon21/LG/dataset/train/label/**/"))

    X_train_dir, X_test_dir, y_train_dir, y_test_dir = train_test_split(
        inputs, labels, test_size=0.15, random_state=1234
    )

    train_dataset = ImageDataset(X_train_dir, y_train_dir, train=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
    )

    train_losses = AverageMeter()
    global_step = 0
    save_psnr_score = 0

    for epoch in tqdm.tqdm(range(args.epochs)):

        progress_bar = tqdm.tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            position=0,
            leave=True,
        )

        for index, batch in progress_bar:
            train_loss = train_on_batch(batch, model, criterion, optimizer, device)
            train_losses.update(train_loss, len(batch[0]))
            global_step += 1

            description = f"EPOCH:[{epoch + 1}] - STEP:[{global_step}] - LOSS:[{train_losses.avg:.4f}]"
            progress_bar.set_description(description)

        with torch.no_grad():

            valid_loss, valid_psnr_score = valid_on_batch(
                X_test_dir, y_test_dir, model, criterion, device, transformer, epoch
            )

            if valid_psnr_score > save_psnr_score:
                save_psnr_score = valid_psnr_score
                torch.save(
                    {
                        "loss": valid_loss,
                        "psnr_score": valid_psnr_score,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                    },
                    os.path.join(
                        "/home/salmon21/LG/WeCanBeLGMan/save_model/effiunet",
                        "checkpoint.pth",
                    ),
                )

        scheduler.step()
        wandb.log(
            {
                "train/lr": scheduler.get_last_lr()[0],
                "train/loss": train_losses.avg,
                "valid/loss": valid_loss,
                "valid/psnr_score": valid_psnr_score,
                "global_steps": global_step,
            }
        )
        train_losses.reset()
        wandb.log({"global_steps": global_step})


if __name__ == "__main__":

    train()
