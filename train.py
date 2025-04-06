from utils import save_logs, get_data, save_img, plot_image
from unet import UNet
from ddpm import Diffusion
from ema import EMA

import torch
from torch.utils.tensorboard import SummaryWriter
from torch import nn, optim
import logging
import os
from tqdm import tqdm
import numpy as np
import argparse
import copy

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

def train(args):
    save_logs(args.run_name)
    device = args.device
    
    dataloader = get_data(args)
    model = UNet().to(device)
    mse = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    diffusion = Diffusion(image_size=args.img_size, device=device)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    size = len(dataloader)
    ema = EMA(beta=0.99)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    
    for epoch in range(args.epoch):
        logging.info(f"Starting epoch {epoch+1}:")
        for i, (image, label) in (pbar := tqdm(dataloader, desc=f"EPOCH:{epoch+1}/{args.epoch}")):
            image = image.to(device)
            label = label.to(device)
            t = diffusion.sample_t(image.shape[0]).to(device)
            x_t, noise = diffusion.noise_sample(image, t)

            if np.random.random() < 0.1:
                label = None
            
            pred_noise = model(x_t, t, label)
            loss = mse(noise, pred_noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step(ema_model, model)
            
            pbar.set_postfix(MSE=loss.item())
            
            logger.add_scalar("MSE", loss.item(), global_step=epoch * size + i)
            
        pred = diffusion.sampling(model, n=image.shape[0])
        save_img(pred, os.path.join("results", args.run_name), f"{epoch}.jpg")
        torch.save(model.state_dict(), os.path.join("models", args.run_name, f"{epoch}_model.pt"))
        
        if epoch % 10 == 0:
            label = torch.arange(10).long().to(device)
            pred = diffusion.sampling(model, n=image.shape[0], labels=label)
            ema_pred = diffusion.sampling(ema_model, n=image.shape[0], labels=label)
            plot_image(pred)
            save_img(pred, os.path.join("results", args.run_name), f"{epoch}_sample.jpg")
            save_img(ema_pred, os.path.join("results", args.run_name), f"{epoch}_ema.jpg")
            torch.save(model.state_dict(), os.path.join("models", args.run_name, f"{epoch}_model.pt"))
            torch.save(ema_model.state_dict(), os.path.join("models", args.run_name, f"{epoch}_ema_model.pt"))
            torch.save(optimizer.state_dict(), os.path.join("models", args.run_name, f"{epoch}_optim.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_Conditional"
    args.epoch = 200
    args.batch_size = 8
    args.img_size = 64
    args.dataset_path = ""
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.lr = 3e-4
    train(args)
