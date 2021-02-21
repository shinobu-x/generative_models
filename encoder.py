import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self, image_size, feature_dim, k, channel):
        super().__init__()
        self.num_layers = 4
        self.num_filters = 32
        self.channel = channel
        self.output_logits = False
        self.feature_dim = feature_dim
        self.k = k
        if image_size == 84:
            self.output_dim = 35
        elif image_size in range(84, 122, 2):
            self.output_dim = 35 + (image_size - 84) // 2
        else:
            raise ValueError(image_size)
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(self.channel, self.num_filters, 3, stride = 2),
            nn.Conv2d(self.num_filters, self.num_filters, 3, stride = 1),
            nn.Conv2d(self.num_filters, self.num_filters, 3, stride = 1),
            nn.Conv2d(self.num_filters, self.num_filters, 3, stride = 1)
            ])
        self.head = nn.Sequential(
            nn.Linear(self.num_filters * self.output_dim ** 2,
                self.feature_dim),
            nn.LayerNorm(self.feature_dim))
        self.outputs = dict()

    def forward(self, observation, detach = False):
        out = self.forward_conv(observation)
        if detach:
            out = out.detach()
        out = self.head(out)
        if not self.output_logits:
            out = torch.tanh(out)
        self.outputs["out"] = out
        return out

    def forward_conv(self, observation):
        observation = observation / 255.0
        self.outputs["observation"] = observation
        out = torch.relu(self.conv_layers[0](observation))
        self.outputs["conv1"] = out
        for i in range(1, self.num_layers):
            out = torch.relu(self.conv_layers[i](out))
            self.outputs["conv%i" % (i + 1)] = out
        out = out.view(out.size(0), -1)
        return out

    def copy_weights(self, source):
        for i in range(self.num_layers):
            self.conv_layers[i].weight = source.conv_layers[i].weight
            self.conv_layers[i].bias = source.conv_layers[i].bias

    def compute_state_entropy(self, source_features, target_features,
            average_entropy = False):
        with torch.no_grad():
            distributions = []
            for i in range(len(target_features) // 10000 + 1):
                start = i * 10000
                end = (i + 1) * 10000
                distribution = torch.norm(source_features[:, None, :] -
                        target_features[None, start:end, :],
                        dim = -1, p = 1)
                distributions.append(dist)
            distributions = torch.cat(distributions, dim = 1)
            knn_distributions = 0.0
            if average_entropy:
                for i in range(5):
                    knn_distributions += torch.kthvalue(distributions, k + 1,
                            dim = 1).value
                    knn_distributions /= 5
            else:
                knn_distributions = torch.kthvalue(distributions, k + 1,
                        dim = 1).value
            state_entropy = knn_distributions
        return state_entropy.unsqueeze(1)
