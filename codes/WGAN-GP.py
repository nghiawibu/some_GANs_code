import torch 
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import torch.nn as nn

from SN import SpectralNorm

def cuda(data):
  if torch.cuda.is_available():
    return data.cuda()
  else:
    return data

def denorm(x):
  out = (x+1)/2
  return out.clamp_(0,1)

fixed_z = cuda(torch.randn(64, 100))
batch_size = 64

# Define data transformer
img_transform = transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])

# Read data and transform
dataset = MNIST(root='./data', download=True, train=True, transform=img_transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

class Generator(nn.Module):
    def __init__(self, batch_size=64, image_size=28, z_dim=100, mlp_dim=64):
        super(Generator, self).__init__()

        self.layers = nn.Sequential(
           nn.Linear(in_features=z_dim, out_features=mlp_dim),
           nn.Linear(in_features=mlp_dim, out_features=mlp_dim),
           nn.Linear(in_features=mlp_dim, out_features=mlp_dim),
           nn.Linear(in_features=mlp_dim, out_features=3)
        )


    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        out = self.layers(z)
        return out



class Critic(nn.Module):
    def __init__(self, batch_size=64, image_size=28, conv_dim=64):
        super(Critic, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=conv_dim, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(conv_dim),
            nn.ReLU(),
            nn.Conv2d(in_channels=conv_dim, out_channels=conv_dim*2, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(conv_dim*2),
            nn.ReLU(),
            nn.Conv2d(in_channels=conv_dim*2, out_channels=conv_dim*4, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(conv_dim*4),
            nn.ReLU(),
            nn.Conv2d(in_channels=conv_dim*4, out_channels=conv_dim*8, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(conv_dim*8),
            nn.ReLU(),
            nn.Conv2d(in_channels=conv_dim*8, out_channels=1, kernel_size=5, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.layers(x)
        return out

def train_WGAN(steps=10000, lr=0.00005, c=0.01, m=64, n_critic=5, z_dim=100):

    G = Generator()
    F = Critic()

    g_optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, G.parameters()),lr=lr)
    f_optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, F.parameters()),lr=lr)

    Iter = iter(dataloader)

    for _ in range(steps):

        try:
            real_images, _ = next(Iter)
        except:
            Iter = iter(dataloader)
            real_images, _ = next(Iter)

        for _ in range(n_critic):
            z = cuda(torch.randn(m, z_dim))
            loss = (F(cuda(real_images)) -F(G(z))).mean()
            f_optimizer.zero_grad(); g_optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(F.parameters(), max_norm=c)
            f_optimizer.step()

        z = cuda(torch.randn(m, z_dim))
        loss = -F(G(z)).mean()
        f_optimizer.zero_grad(); g_optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(G.parameters, max_norm=c)
        g_optimizer.step()
    
def train_WGAN_GP(steps=10000, alpha=0.0001, beta_1=0, beta_2=0.9, m=64, n_critic=5, z_dim=100, penalty_const=10):

    G = Generator()
    D = Critic()

    g_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, G.parameters()),lr=alpha, betas={beta_1,beta_2})
    d_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, D.parameters()),lr=alpha, betas={beta_1,beta_2})

    Iter = iter(dataloader)

    for _ in range(steps):
        try:
            real_images, _ = next(Iter)
        except:
            Iter = iter(dataloader)
            real_images, _ = next(Iter)
        for _ in range(n_critic):
            losses = []
            for x in real_images:
                eps = torch.rand()
                z = cuda(torch.randn(1, z_dim))
                x_tilde = G(z)
                x_hat = torch.tensor(eps*x + (1-eps)*x_tilde, requires_grad=True)
                D_hat = D(x_hat)
                D_hat.backward()
                dx = x_hat.grad

                losses.append(D(x_tilde)-D(x)+penalty_const*((dx.norm(dim=0,p=2)-1)**2))
            loss = losses.mean()
            loss.backward()

            d_optimizer.zero_grad()
            d_optimizer.step()

        z = cuda(torch.randn(m, z_dim))
        loss = -D(G(z)).mean()
        loss.backward()

        g_optimizer.zero_grad()
        g_optimizer.step()





        

        