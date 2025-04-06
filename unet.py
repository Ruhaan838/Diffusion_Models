import torch
from torch import nn
from torch.nn import functional as F

class FeedForward(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.l1 = nn.Linear(in_channels, out_channels)
        self.l2 = nn.Linear(out_channels, in_channels)
    
    def forward(self, x):
        return self.l2(F.gelu(self.l1(x)))


class SelfAttention(nn.Module):
    def __init__(self, in_channels, im_size, dropout=0.1):
        super().__init__()
        
        self.in_channels = in_channels
        self.im_size = im_size
        
        self.mla = nn.MultiheadAttention(in_channels, 4, batch_first=True)
        self.attention_norm = nn.LayerNorm(in_channels)
        
        self.ff = FeedForward(in_channels, in_channels)
        self.ff_norm = nn.LayerNorm(in_channels)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = x.view(-1, self.in_channels, self.im_size * self.im_size).transpose(1, 2)
        x_ = self.attention_norm(x)
        attention, _ = self.mla(x_, x_, x_)
        x = x + self.dropout(attention)
        x = self.ff_norm(x)
        x = x + self.ff(x)
        return x.transpose(1, 2).view(-1, self.in_channels, self.im_size, self.im_size)
 
 
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, min_channels=None, residual=False):
        super().__init__()
        
        self.residual = residual
        if not min_channels:
            min_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, min_channels, kernel_size=3, padding=1, bias=False)
        self.conv1_gn = nn.GroupNorm(1, min_channels)
        self.conv2 = nn.Conv2d(min_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.conv2_gn = nn.GroupNorm(1, out_channels)

        
    def forward(self, x):
        x_ = self.conv1(x)
        x_ = self.conv1_gn(x_)
        x_ = F.gelu(x_)
        x_ = self.conv2(x_)
        x_ = self.conv2_gn(x_)
        
        if self.residual:
            return F.gelu(x + x_)
        
        return x_

class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2)
        self.conv_1 = DoubleConv(in_channels, in_channels, residual=True)
        self.conv_2 = DoubleConv(in_channels, out_channels)
        
        self.emb = nn.Linear(emb_dim, out_channels)

    def forward(self, x, t):
        x = self.max_pool_1(x)
        x = self.conv_1(x)
        x = self.conv_2(x)
        emb = F.silu(self.emb(t))[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb
    
    
class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, emb_size=256):
        super().__init__()
        
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv1 = DoubleConv(in_channels, in_channels, residual=True)
        self.conv2 = DoubleConv(in_channels, out_channels, in_channels // 2)
        
        self.emb = nn.Linear(emb_size, out_channels)
        
    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv2(self.conv1(x))
        emb = F.silu(self.emb(t))[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, time_dim=256, num_classes=None, device="cpu"):
        super().__init__()
        
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(in_channels, 64)
        
        self.down1 = DownConv(64, 128)
        self.att1 = SelfAttention(128, 32)
        
        self.down2 = DownConv(128, 256)
        self.att2 = SelfAttention(256, 16)
        
        self.down3 = DownConv(256, 256)
        self.att3 = SelfAttention(256, 8)
        
        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)
        
        self.up1 = UpConv(512, 128)
        self.att4 = SelfAttention(128, 16)
        
        self.up2 = UpConv(256, 64)
        self.att5 = SelfAttention(64, 32)
        
        self.up3 = UpConv(128, 64)
        self.att6 = SelfAttention(64, 64)
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)
        
        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)
        
    def pos_emgedding(self, t, chennels):
        div_term = (10000 ** (torch.arange(0, chennels, 2, device=self.device).float() / chennels))
        
        sin_emb = torch.sin(t.repeat(1, chennels // 2) / div_term)
        cos_emb = torch.cos(t.repeat(1, chennels // 2) / div_term)
        pos_enc = torch.cat([sin_emb, cos_emb], dim=-1)
        return pos_enc
    
    def forward(self, x, t, y=None):
        """
            x: noise image
            t: time step
        """
        
        t = t.unsqueeze(-1).to(torch.float)
        t = self.pos_emgedding(t, self.time_dim)
        
        if y is not None:
            t += self.label_emb(y)
        
        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.att1(x2)
        
        x3 = self.down2(x2, t)
        x3 = self.att2(x3)
        
        x4 = self.down3(x3, t)
        x4 = self.att3(x4)
        
        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)
        
        x = self.up1(x4, x3, t)
        x = self.att4(x)
        
        x = self.up2(x, x2, t)
        x = self.att5(x)
        
        x = self.up3(x, x1, t)
        x = self.att6(x)
        
        return self.outc(x)
        
        
if __name__ == "__main__":
    
    model = UNet(3, 3, num_classes=10)        
    x = torch.randn(1, 3, 64, 64)
    t = torch.tensor([500] * x.shape[0]).long()
    y = torch.tensor([1] * x.shape[0]).long()
    print(model(x, t, y).shape)
