import torch
#torch.cuda.current_device()
import torch.nn as nn
from torchvision import transforms


class VAE(nn.Module):
    """VAE for 64x64 face generation.

    The hidden dimensions can be tuned.
    """

    def __init__(self, hiddens=[16, 32, 64, 128, 256], latent_dim=128) -> None:
        super().__init__()

        # encoder

        prev_channels = 1
        modules = []
        img_length = 416
        for cur_channels in hiddens:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(prev_channels,
                              cur_channels,
                              kernel_size=3,
                              stride=2,
                              padding=1), nn.BatchNorm2d(cur_channels),
                    nn.ReLU()))
            prev_channels = cur_channels
            img_length //= 2
        self.encoder = nn.Sequential(*modules)

        print(prev_channels)
        print(img_length)
        self.mean_linear = nn.Linear(prev_channels * (img_length+1)*(img_length+1),
                                     latent_dim)
        self.var_linear = nn.Linear(prev_channels * (img_length+1)*(img_length+1),
                                    latent_dim)
        self.latent_dim = latent_dim
        # decoder
        modules = []
        self.decoder_projection = nn.Linear(
            latent_dim, prev_channels * img_length * img_length)
        self.decoder_input_chw = (prev_channels, img_length, img_length)

        for i in range(len(hiddens) - 1, 0, -1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hiddens[i],
                                       hiddens[i - 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hiddens[i - 1]), nn.ReLU()))

        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hiddens[0],
                                   hiddens[0],
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   output_padding=1),
                nn.BatchNorm2d(hiddens[0]), nn.ReLU(),
                nn.Conv2d(hiddens[0], 1, kernel_size=3, stride=1, padding=1),
                nn.ReLU()))
        self.decoder = nn.Sequential(*modules)
        self.final_process = transforms.CenterCrop(399)
        self.label = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 1), 
            nn.ReLU()
        )
        

    def forward(self, x):
        x = torch.nn.functional.pad(x, (17, 17, 17, 17))
        encoded = self.encoder(x)
        encoded = torch.flatten(encoded, 1)
        print(encoded.shape)
        mean = self.mean_linear(encoded)
        print(encoded.shape)
        logvar = self.var_linear(encoded)
        eps = torch.randn_like(logvar)
        std = torch.exp(logvar / 2)
        z = eps * std + mean
        y = self.label(z)
        x = self.decoder_projection(z)
        x = torch.reshape(x, (-1, *self.decoder_input_chw))
        decoded = self.final_process(self.decoder(x))
        


        return decoded, y, mean, logvar

    def sample(self, device='cpu'):
        z = torch.randn(1, self.latent_dim).to(device)
        x = self.decoder_projection(z)
        x = torch.reshape(x, (-1, *self.decoder_input_chw))
        decoded = self.final_process(self.decoder(x))
        label = self.label(z)
        return decoded, label
    '''
class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.mean_linear1 = nn.Linear(79800, 2*latent_dim)
        self.mean_linear2 = nn.Linear(2*latent_dim, latent_dim)
        self.var_linear1 = nn.Linear(79800, 2*latent_dim)
        self.var_linear2 = nn.Linear(2*latent_dim, latent_dim)
        self.decoder_projection1 = nn.Linear(latent_dim, 5*latent_dim)
        self.decoder_projection2 = nn.Linear(5*latent_dim, 79800)    
        self.relu = nn.ReLU()

    def forward(self, x):
        mean = self.mean_linear2(self.relu(self.mean_linear1(x)))
        logvar = self.var_linear2(self.relu(self.var_linear1(x)))
        eps = torch.randn_like(logvar)
        std = torch.exp(logvar / 2)
        z = eps * std + mean
        x = self.decoder_projection2(self.relu(self.decoder_projection1(z)))
        return x, mean, logvar
    '''