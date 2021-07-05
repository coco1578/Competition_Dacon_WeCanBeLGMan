import os
import glob

import tqdm
import wandb
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms

from sklearn.model_selection import train_test_split
from torch.utils.data.dataloader import DataLoader

from dataset import ImageDataset
from CBDNet.model.cbdnet import Network, fixed_loss
from utils import recover_image, psnr_score, AverageMeter


def train_on_batch(batch, model, criterion, optimizer, device):

    model.train()

    images = batch[0].to(device)
    labels = batch[1].to(device)

    noise_level_est, outputs = model(images)
    loss = criterion(outputs, labels, noise_level_est, 0, 0)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.data.cpu().item()


def valid_on_batch(X_test_dir, y_test_dir, model, criterion, device, transformer):

    valid_losses = []
    valid_psnr_scores = []

    model.eval()

    for X_tes_dir, y_tes_dir in tqdm.tqdm(zip(X_test_dir, y_test_dir)):
        print(X_tes_dir, y_tes_dir)
        valid_loss = []
        top_lefts = np.load(os.path.join(X_tes_dir, "top_lefts", "top_lefts.npy"))

        result_X_image = np.zeros((top_lefts[-1][0] + 204, top_lefts[-1][1] + 204, 3))
        result_X_masks = np.zeros((top_lefts[-1][0] + 204, top_lefts[-1][1] + 204, 3))

        result_y_image = np.zeros((top_lefts[-1][0] + 204, top_lefts[-1][1] + 204, 3))
        result_y_masks = np.zeros((top_lefts[-1][0] + 204, top_lefts[-1][1] + 204, 3))

        for i in range(len(top_lefts)):

            X_tes = np.load(os.path.join(X_tes_dir, f"{i}.npy"))
            y_tes = np.load(os.path.join(y_tes_dir, f"{i}.npy"))

            X_tes = transformer(X_tes)
            X_tes = X_tes.unsqueeze(dim=0)
            X_tes = X_tes.to(device)

            noise_level_est, output = model(X_tes)
            loss = criterion(
                output,
                transformer(y_tes).unsqueeze(dim=0).to(device),
                noise_level_est,
                0,
                0,
            )

            output = output.cpu().detach().squeeze().permute(1, 2, 0).numpy()
            output = np.uint8(output * 255.0)

            result_X_image[
                top_lefts[i][0] : top_lefts[i][0] + 204,
                top_lefts[i][1] : top_lefts[i][1] + 204,
                :,
            ] += output
            result_X_masks[
                top_lefts[i][0] : top_lefts[i][0] + 204,
                top_lefts[i][1] : top_lefts[i][1] + 204,
                :,
            ] += 1

            result_y_image[
                top_lefts[i][0] : top_lefts[i][0] + 204,
                top_lefts[i][1] : top_lefts[i][1] + 204,
                :,
            ] += y_tes
            result_y_masks[
                top_lefts[i][0] : top_lefts[i][0] + 204,
                top_lefts[i][1] : top_lefts[i][1] + 204,
                :,
            ] += 1

            valid_loss.append(loss.data.cpu().item())

        result_X_image = result_X_image / result_X_masks
        result_y_image = result_y_image / result_y_masks

        result_X_image = np.uint8(result_X_image)
        result_y_image = np.uint8(result_y_image)

        valid_losses.append(np.mean(valid_loss))
        valid_psnr_scores.append(
            psnr_score(result_X_image.astype(float), result_y_image.astype(float))
        )

    return np.mean(valid_losses), np.mean(valid_psnr_scores)


def train():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    wandb.login()
    wandb.init(project="Dacon", entity="coco1578", name="CBDNet")

    model = Network()
    model_info = torch.load(
        "/home/salmon21/LG/WeCanBeLGMan/save_model/cbdnet/checkpoint.pth.tar",
        map_location="cpu",
    )
    model.load_state_dict(model_info["state_dict"], strict=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=10, eta_min=1e-6
    )
    transformer = transforms.Compose([transforms.ToTensor()])

    criterion = fixed_loss()

    model, criterion = model.to(device), criterion.to(device)

    inputs = sorted(glob.glob("/home/salmon21/LG/dataset/train/input/**/"))
    labels = sorted(glob.glob("/home/salmon21/LG/dataset/train/label/**/"))

    X_train_dir, X_test_dir, y_train_dir, y_test_dir = train_test_split(
        inputs, labels, test_size=0.15, random_state=5252
    )

    train_dataset = ImageDataset(X_train_dir, y_train_dir, train=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
    )

    train_losses = AverageMeter()
    max_epochs = 300
    global_step = 0
    save_psnr_score = 0

    for epoch in tqdm.tqdm(range(max_epochs)):

        progress_bar = tqdm.tqdm(
            enumerate(train_loader),
            total=int(len(train_loader) / 128),
            position=0,
            leave=True,
        )

        for index, batch in progress_bar:
            train_loss = train_on_batch(batch, model, criterion, optimizer, device)
            train_losses.update(train_loss, len(batch[0]))
            global_step += 1

            description = f"EPOCH:[{epoch + 1}] - STEP:[{global_step}] - LOSS:[{train_losses.avg:.4f}]"
            progress_bar.set_description(description)

            if global_step % 500 == 0:
                with torch.no_grad():

                    valid_loss, valid_psnr_score = valid_on_batch(
                        X_test_dir, y_test_dir, model, criterion, device, transformer
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
                                "/home/salmon21/LG/WeCanBeLGMan/save_model/cbdnet",
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

            else:
                wandb.log({"global_steps": global_step})


if __name__ == "__main__":

    train()
