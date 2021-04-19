import os
import pickle
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
from torchvision.utils import save_image


class Generator(nn.Module):
    def __init__(self, input_channels, n_features, output_channels, pooling_kernels, pooling_strides, pooling_pads):
        super().__init__()
        self.m1 = self.Module(input_channels, n_features*8, kernel=pooling_kernels[0], stride=pooling_strides[0], pad=pooling_pads[0])
        self.m2 = self.Module(n_features*8, n_features*4, kernel=pooling_kernels[1], stride=pooling_strides[1], pad=pooling_pads[1])
        self.m3 = self.Module(n_features*4, n_features*2, kernel=pooling_kernels[2], stride=pooling_strides[2], pad=pooling_pads[2])
        self.m4 = self.Module(n_features*2, n_features, kernel=pooling_kernels[3], stride=pooling_strides[3], pad=pooling_pads[3])
        self.bottle = self.Module(n_features, output_channels, kernel=pooling_kernels[4], stride=pooling_strides[4], pad=pooling_pads[4], bn=False, activation='tanh')

    def forward(self, x):
        return self.bottle(self.m4(self.m3(self.m2(self.m1(x)))))

    class Module(nn.Module):
        def __init__(self, input_channels, output_channels, kernel, stride, pad, bn=True, activation='relu'):
            super().__init__()
            self.convt = nn.ConvTranspose2d(input_channels, output_channels, kernel_size=kernel, stride=stride, padding=pad)
            if bn:
                self.bn = nn.BatchNorm2d(output_channels)
            else:
                self.bn = None
            if activation == 'relu':
                self.activation = nn.ReLU(inplace=True)
            elif activation == 'tanh':
                self.activation = nn.Tanh()

        def forward(self, x):
            if self.bn is not None:
                return self.activation(self.bn(self.convt(x)))
            else:
                return self.activation(self.convt(x))


class Discriminator(nn.Module):
    def __init__(self, input_channels, n_features, output_channels, pooling_kernels, pooling_strides, pooling_pads):
        super().__init__()
        self.m1 = self.Module(input_channels, n_features, kernel=pooling_kernels[0], stride=pooling_strides[0], pad=pooling_pads[0], bn=False)
        self.m2 = self.Module(n_features, n_features*2, kernel=pooling_kernels[1], stride=pooling_strides[1], pad=pooling_pads[1])
        self.m3 = self.Module(n_features*2, n_features*4, kernel=pooling_kernels[2], stride=pooling_strides[2], pad=pooling_pads[2])
        self.m4 = self.Module(n_features*4, n_features*8, kernel=pooling_kernels[3], stride=pooling_strides[3], pad=pooling_pads[3])
        self.bottle = self.Module(n_features*8, output_channels, kernel=pooling_kernels[4], stride=pooling_strides[4], pad=pooling_pads[4], bn=False, activation='sigmoid')

    def forward(self, x):
        return self.bottle(self.m4(self.m3(self.m2(self.m1(x)))))

    class Module(nn.Module):
        def __init__(self, input_channels, output_channels, kernel, stride, pad, bn=True, activation='leaky_relu'):
            super().__init__()
            self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=kernel, stride=stride, padding=pad)
            if bn:
                self.bn = nn.BatchNorm2d(output_channels)
            else:
                self.bn = None
            if activation == 'leaky_relu':
                self.activation = nn.LeakyReLU(0.2, inplace=True)
            elif activation == 'sigmoid':
                self.activation = nn.Sigmoid()

        def forward(self, x):
            if self.bn is not None:
                return self.activation(self.bn(self.conv(x)))
            else:
                return self.activation(self.conv(x))


class NetManager(nn.Module):
    def __init__(self, model_name, input_dir, output_dir):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        super().__init__()

        # param
        image_size = 64
        self.batch_size = 128
        self.latent_dim = 128

        g_param = {}
        g_param['input_channels'] = self.latent_dim
        g_param['n_features'] = 64
        g_param['output_channels'] = 3
        g_param['pooling_kernels'] = [4, 4, 4, 4, 4]
        g_param['pooling_strides'] = [1, 2, 2, 2, 2]
        g_param['pooling_pads'] = [0, 1, 1, 1, 1]

        d_param = {}
        d_param['input_channels'] = 3
        d_param['n_features'] = 64
        d_param['output_channels'] = 1
        d_param['pooling_kernels'] = [4, 4, 4, 4, 4]
        d_param['pooling_strides'] = [2, 2, 2, 2, 2]
        d_param['pooling_pads'] = [1, 1, 1, 1, 0]

        # Generator
        self.G = Generator(**g_param).to(self.device)
        self.G.apply(self.weights_init)

        # Discriminator
        self.D = Discriminator(**d_param).to(self.device)
        self.D.apply(self.weights_init)

        # model name
        self.model_name = model_name

        # output_dir
        self.output_dir = f'{output_dir}/{self.model_name}'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        print(f'{self.output_dir} is output directory')

        # history
        self.history_path = f'{self.output_dir}/history.dat'

        # model
        self.model_path = f'{self.output_dir}/model.pth'

        # label
        self.real_label = 1
        self.fake_label = 0

        # dataloader
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        download_dir = "../data"
        dataset = datasets.CelebA(
            download_dir, transform=transform, download=True
        )
        train_size = int(len(dataset) * 0.8)
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)

    def init_model(self, ):
        lr = 0.0002
        beta1 = 0.5

        self.load_history()
        self.load_model()
        self.g_optimizer = optim.Adam(self.G.parameters(), lr=lr, betas=(beta1, 0.999))
        self.d_optimizer = optim.Adam(self.D.parameters(), lr=lr, betas=(beta1, 0.999))
        self.criterion = nn.BCELoss()
        self.fixed_z = torch.randn(64, self.latent_dim, 1, 1, device=self.device)

    def fina_model(self):
        self.save_model()
        self.save_history()

    def g_train(self):
        #self.G.zero_grad()
        self.g_optimizer.zero_grad()
        z = torch.randn(self.batch_size, self.latent_dim, 1, 1, device=self.device)
        y_pred = self.D(self.G(z))
        y = torch.full_like(y_pred, self.real_label)
        loss = self.criterion(y_pred, y)
        loss.backward()
        self.g_optimizer.step()
        return float(loss)

    def generate_image(self):
        z = torch.randn(self.batch_size, self.latent_dim, 1, 1, device=self.device)
        image = self.G(z)
        return image

    def d_train(self, x_real):
        #self.D.zero_grad()
        self.d_optimizer.zero_grad()
        y_pred = self.D(x_real)
        y_real = torch.full_like(y_pred, self.real_label)
        loss_real = self.criterion(y_pred, y_real)
        z = torch.randn(self.batch_size, self.latent_dim, 1, 1, device=self.device)
        y_pred = self.D(self.G(z))
        y_fake = torch.full_like(y_pred, self.fake_label)
        loss_fake = self.criterion(y_pred, y_fake)
        loss = loss_real + loss_fake
        loss.backward()
        self.d_optimizer.step()
        return float(loss)

    def train(self):
        self.G.train()
        self.D.train()

        print(f'\nEpoch: {self.history["epoch"]+1:d}')

        d_loss = 0
        g_loss = 0
        samples_cnt = 0
        for batch_idx, (x, _) in enumerate(self.train_dataloader):
            x = x.to(self.device)
            d_loss += self.d_train(x)
            g_loss += self.g_train()
            samples_cnt += x.size(0)
            if batch_idx%10 == 0:
                print(batch_idx, len(self.train_dataloader), f'd_loss: {d_loss/samples_cnt:f}', f'g_loss: {g_loss/samples_cnt:f}')
        self.history['epoch'] += 1
        self.history['d_loss'].append(d_loss/samples_cnt)
        self.history['g_loss'].append(g_loss/samples_cnt)

    def test(self):
        self.G.eval()
        self.D.eval()

        with torch.no_grad():
            image = self.generate_image()
            save_image(image, f'{self.output_dir}/output_epoch_{str(self.history["epoch"])}.png', nrow=8)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def load_history(self):
        if os.path.exists(self.history_path):
            with open(self.history_path, 'rb') as fp:
                self.history = pickle.load(fp)
            print(f'{self.history_path} is loaded')
        else:
            self.history = {'epoch':0, 'g_loss':[], 'd_loss':[]}
            print(f'{self.history_path} is not found')

    def save_history(self):
        with open(self.history_path, "wb") as fp:
            pickle.dump(self.history, fp)
        print(f'{self.history_path} is saved')

    def load_model(self):
        if os.path.exists(self.model_path):
            self.load_state_dict(torch.load(self.model_path))
            print(f'{self.model_path} is loaded')
        else:
            print(f'{self.model_path} is not found') 

    def save_model(self):
        torch.save(self.state_dict(), self.model_path)
        print(f'{self.model_path} is saved')
