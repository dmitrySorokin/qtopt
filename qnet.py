import torch
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size=512):
        super(QNetwork, self).__init__()

        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.
                constant_(x, 0), nn.init.calculate_gain('relu')
        )

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)), nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), Flatten(),
            init_(nn.Linear(2048, hidden_size)), nn.ReLU()
        )

        self.linear1 = nn.Sequential(
            init_(nn.Linear(hidden_size + num_outputs, 512)), nn.ReLU()
        )

        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.
                constant_(x, 0)
        )

        self.linear2 = init_(nn.Linear(512, 1))

    def forward(self, inputs, action):
        inputs = inputs.float()
        x = inputs / 255
        x = self.main(x)

        x = torch.cat([x, action], 1)
        x = self.linear1(x)
        x = self.linear2(x)

        return x
