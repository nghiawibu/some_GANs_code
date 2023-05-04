import torch 
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as T
import torch.nn as nn
import cv2
import albumentations as A
import os
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
from torchsummary import summary
# train_dataset = YOLODataset(
#     train_csv_path,
#     transform=config.train_transforms,
#     S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
#     img_dir=config.IMG_DIR,
#     label_dir=config.LABEL_DIR,
#     anchors=config.ANCHORS,
# )
# train_loader = DataLoader(
#     dataset=train_dataset,
#     batch_size=config.BATCH_SIZE,
#     num_workers=config.NUM_WORKERS,
#     pin_memory=config.PIN_MEMORY,
#     shuffle=True,
#     drop_last=False,
# )


class P2PDataset(Dataset):
    def __init__(self,data_dir,csv_file,transform=None):
        self.annotations = pd.read_csv(data_dir+csv_file)
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        input_path = os.path.join(self.data_dir, self.annotations.iloc[index, 1])
        input_im = np.array(Image.open(input_path).convert("RGB"))
        real_path = os.path.join(self.data_dir, self.annotations.iloc[index, 2])
        real_im = np.array(Image.open(real_path).convert("RGB"))

        if self.transform:
            input_im, real_im = self.transform([input_im, real_im])

        return input_im, real_im
    
IMAGE_SIZE = 256
img_transform = T.Compose([
    T.Resize(size=(286,286)),
    T.RandomCrop(size=(IMAGE_SIZE,IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,))
])

batch_size=1
dataset = P2PDataset(data_dir="./data/pix2pix/maps/maps/train_processed/", csv_file="data.csv", transform=img_transform)
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)



def cuda(data):
  if torch.cuda.is_available():
    return data.cuda()
  else:
    return data





# Read data and transform
# dataset = MNIST(root='./data', download=True, train=True, transform=img_transform)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

class EncoderBlock(nn.Module):
    def __init__(self, in_dim, out_dim, batch_norm=True, output=False):
        super(EncoderBlock, self).__init__()
        layers = []
        self.output=output

        conv1 = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=4, stride=2, padding=1)
        nn.init.normal_(conv1.weight, mean=0, std=0.02)
        layers.append(conv1) 

        if batch_norm: 
            layers.append(nn.BatchNorm2d(num_features=out_dim))

        if output:
            layers.append(nn.Sigmoid())
        else:    
            layers.append(nn.LeakyReLU(0.2))

        self.layers = nn.Sequential(*layers)
    def forward(self,x):
        out = self.layers(x)
        if self.output:
            out = torch.mean(out, (1,2,3))
        return out
    
class DecoderBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=True):
        super(DecoderBlock, self).__init__()
        layers = []
        conv1 = nn.ConvTranspose2d(in_channels=in_dim, out_channels=out_dim, kernel_size=4, stride=2, padding=1)
        nn.init.normal_(conv1.weight, mean=0, std=0.02) 
        layers.append(conv1)  
        layers.append(nn.BatchNorm2d(num_features=out_dim))

        if dropout:
            layers.append(nn.Dropout(0.5))
        layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)
    def forward(self,x, skip_activation):
        out = self.layers(x)
        # print(out.shape)
        # print(skip_activation.shape)
        # return
        out = torch.cat((out, skip_activation), dim=1)
        return out

class UNetGenerator(nn.Module):
    def __init__(self,in_dim=103, conv_dim=64):
        super(UNetGenerator, self).__init__()

        #C64-C128-C256-C512-C512-C512-C512-C512
        self.encoder1 = EncoderBlock(in_dim, conv_dim, batch_norm=False)
        self.encoder2 = EncoderBlock(conv_dim, conv_dim*2)
        self.encoder3 = EncoderBlock(conv_dim*2, conv_dim*4)
        self.encoder4 = EncoderBlock(conv_dim*4, conv_dim*8)
        self.encoder5 = EncoderBlock(conv_dim*8, conv_dim*8)
        self.encoder6 = EncoderBlock(conv_dim*8, conv_dim*8)
        self.encoder7 = EncoderBlock(conv_dim*8, conv_dim*8)
        self.encoder8 = EncoderBlock(conv_dim*8, conv_dim*8)

        #CD512-CD1024-CD1024-C1024-C1024-C512-C256-C128
        self.decoder8 = DecoderBlock(conv_dim*8, conv_dim*8)
        self.decoder7 = DecoderBlock(conv_dim*16, conv_dim*16)
        self.decoder6 = DecoderBlock(conv_dim*24, conv_dim*16)
        self.decoder5 = DecoderBlock(conv_dim*24, conv_dim*16, dropout=False)
        self.decoder4 = DecoderBlock(conv_dim*24, conv_dim*16, dropout=False)
        self.decoder3 = DecoderBlock(conv_dim*20, conv_dim*8, dropout=False)
        self.decoder2 = DecoderBlock(conv_dim*10, conv_dim*4, dropout=False)
        self.decoder1 = DecoderBlock(conv_dim*5, conv_dim*2, dropout=False)
        layer = nn.Conv2d(in_channels=conv_dim*2+in_dim, out_channels=3, kernel_size=4, stride=2, padding=1)
        nn.init.normal_(layer.weight, mean=0, std=0.02)
        self.final_layers = nn.Sequential(
            layer,
            nn.Tanh()
        )

    def forward(self, x, z):
        x = torch.cat((x, z), dim=3)
        x = x.permute(0,3,1,2)
        outs = [x]
        out = self.encoder1(x); outs.append(out)
        out = self.encoder2(out); outs.append(out)
        out = self.encoder3(out); outs.append(out)
        out = self.encoder4(out); outs.append(out)
        out = self.encoder5(out); outs.append(out)
        out = self.encoder6(out); outs.append(out)
        out = self.encoder7(out); outs.append(out) 
        out = self.encoder8(out); outs.append(out)

        out = self.decoder8(out, outs[7])
        out = self.decoder7(out, outs[6])
        out = self.decoder6(out, outs[5])
        out = self.decoder5(out, outs[4])
        out = self.decoder4(out, outs[3])
        out = self.decoder3(out, outs[2])
        out = self.decoder2(out, outs[1])
        out = self.decoder1(out, outs[0])
        out = self.final_layers(out)
        return out

class Discriminator(nn.Module):
    def __init__(self, in_dim=6, conv_dim=64):
        super(Discriminator, self).__init__()
        #C64-C128-C256-C512
        self.layers = nn.Sequential(
            EncoderBlock(in_dim, conv_dim, batch_norm=False),
            EncoderBlock(conv_dim, conv_dim*2),
            EncoderBlock(conv_dim*2, conv_dim*4),
            EncoderBlock(conv_dim*4, conv_dim*8),
            EncoderBlock(conv_dim*8, 1, batch_norm=False, output=True)
        )
    def forward(self, x, y):
        x = torch.cat((x, y), dim=3)
        x = x.permute(0,3,1,2)
        out = self.layers(x)
        return out
               

class pix2pix(nn.Module):
    def __init__(self):
        super(pix2pix, self).__init__()
        self.generator = UNetGenerator()
        self.discriminator = Discriminator()

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location='cuda')
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

#real_image, input_image, fake_image
def train(steps, alpha, beta_1, beta_2, batch_size, z_dim=100):
    G = UNetGenerator()
    D = Discriminator()
    g_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, G.parameters()),lr=alpha, betas={beta_1,beta_2})
    d_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, D.parameters()),lr=alpha, betas={beta_1,beta_2})
    Iter = iter(dataloader)
    for step in range(steps):
        
        # load data
        try:
            input_im, real_im = next(Iter)
        except:
            Iter = iter(dataloader)
            input_im, real_im = next(Iter)
        _, H, W, _ = input_im.size
        input_im = cuda(input_im)
        real_im = cuda(real_im)

        # training discriminator
        z = torch.randn(batch_size,H,W,z_dim)
        fake_im = G(input_im, z)
        loss_GAN = (torch.log(D(input_im,real_im))+torch.log(1-D(input_im, fake_im))).mean()
        loss_L1 = torch.mean(torch.norm(real_im-fake_im, p=1, dim=0))
        loss = loss_GAN+loss_L1
        d_optimizer.zero_grad()
        loss.backward()
        d_optimizer.step()    

        #training generator
        z = torch.randn(batch_size,H,W,z_dim)
        loss_G = -torch.log(D(input_im, fake_im)).mean()
        g_optimizer.zero_grad()
        loss_G.backward()
        g_optimizer.step()

        #save parameters
        if (step+1)%10==0:
            save_checkpoint(G, g_optimizer, "./checkpoints/pix2pix/generator/cp.pth.tar")
            save_checkpoint(D, d_optimizer, "./checkpoints/pix2pix/discriminator/cp.pth.tar")

if __name__ == "__main__":
    G = UNetGenerator()
    summary(G.cuda(), input_size=[(256,256,3),(256,256,100)])
    # train(30, 0.0002, 0.5, 0.999, batch_size)
    #infer and save generated images
    

