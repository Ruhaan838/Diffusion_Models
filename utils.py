import torch
import torchvision
from torchvision import transforms as T
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os


def plot_image(image):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in image.cpu()], dim=-1)
        
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()
    
def save_img(image, path, name, **kwargs):
    os.makedirs(path, exist_ok=True)
    grid = torchvision.utils.make_grid(image, **kwargs)
    ndarr = grid.permute(1, 2, 0).to("cpu").numpy()
    im = Image.fromarray((ndarr * 255).astype('uint8'))
    im.save(os.path.join(path, name))

def get_data(args):
    transforms = T.Compose([
        T.Resize(80),
        T.RandomResizedCrop(args.img_size, scale=(0.8, 1.0)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    return dataloader

def save_logs(run_name):
    os.makedirs("models", exist_ok=True)
    # Fix: Correct typo in directory name
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)

