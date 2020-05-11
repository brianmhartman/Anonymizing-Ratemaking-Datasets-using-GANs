import torch.nn as nn


def init_weights_relu(module):
    """Initialize weights for linear layers when activation is LeakyReLu."""
    if isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight.data,
                                 nonlinearity="leaky_relu")
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0.01)


def init_weights_tanh(module):
    """Initialize weights for linear layers when activation is tanh."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight.data)
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0.01)


class Autoencoder(nn.Module):
    def __init__(self, example_dim, compression_dim,
                 feature_choice=None, binary=True,
                 device='cpu', init_weights=True):
        super(Autoencoder, self).__init__()

        self.compression_dim = compression_dim
        self.feature_choice = feature_choice
        self.activation = "tanh" if binary else "leaky_relu"

        self.encoder = nn.Sequential(
            nn.Linear(example_dim, (example_dim + compression_dim) // 2),
            nn.Tanh() if binary else nn.LeakyReLU(0.2),
            nn.Linear((example_dim + compression_dim) // 2, compression_dim),
            nn.Tanh() if binary else nn.LeakyReLU(0.2)
        ).to(device)

        self.decoder = nn.Sequential(
            nn.Linear(compression_dim, (example_dim + compression_dim) // 2),
            nn.Tanh() if binary else nn.LeakyReLU(0.2),
            nn.Linear((example_dim + compression_dim) // 2, example_dim),
            nn.Sigmoid()
        ).to(device)

        # Initialize weights if desired
        if init_weights:
            if self.activation == "leaky_relu":
                self.apply(init_weights_relu)
            else:
                self.apply(init_weights_tanh)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def get_compression_dim(self):
        return self.compression_dim
