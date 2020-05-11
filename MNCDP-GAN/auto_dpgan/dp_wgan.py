import torch.nn as nn

from auto_dpgan.dp_autoencoder import init_weights_relu, init_weights_tanh


class Generator(nn.Module):
    def __init__(self, input_dim, output_dim,
                 binary=True, device='cpu', init_weights=True):
        super(Generator, self).__init__()

        self.latent_dim = input_dim  # Store latent dimension
        # A lambda can't be saved as a model parameter so it has to be a var
        activation_fn = nn.Tanh if binary else lambda: nn.LeakyReLU(0.2)
        # WGAN recommended LeakyReLu slope is 0.2

        def block(inp, out, activation, block_device):
            return nn.Sequential(
                nn.Linear(inp, out, bias=False),
                nn.LayerNorm(out),  # Recommended by Gulrajani et al 2017
                activation(),
            ).to(block_device)

        self.block_0 = block(input_dim, input_dim, activation_fn, device)
        self.block_1 = block(input_dim, input_dim, activation_fn, device)
        self.block_2 = block(input_dim, output_dim, activation_fn, device)

        # Initialize weights if desired
        if init_weights:
            if isinstance(activation_fn, nn.Tanh):
                self.apply(init_weights_tanh)
            else:
                self.apply(init_weights_relu)

    def forward(self, x):
        x = self.block_0(x) + x
        x = self.block_1(x) + x
        x = self.block_2(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, input_dim, device='cpu', init_weights=True):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, (2 * input_dim) // 3),
            nn.LeakyReLU(0.2),
            nn.Linear((2 * input_dim) // 3, input_dim // 3),
            nn.LeakyReLU(0.2),
            nn.Linear(input_dim // 3, 1),
        ).to(device)

        # Initialize weights if desired
        if init_weights:
            self.apply(init_weights_relu)

    def forward(self, x):
        return self.model(x)
