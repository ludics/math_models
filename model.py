import torch
from torch import nn

class Block1(nn.Module):
    def __init__(self, inplanes=16, outplanes=16, kernel_size=[1, 3], padding=[0, 1]):
        super().__init__()
        self.bn = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        identity = x
        out = self.conv(self.relu(self.bn(x)))
        out += identity
        out = self.relu(out)
        return out

class Block2(nn.Module):
    def __init__(self, inplanes=320, outplanes=320, kernel_size=3, stride=2):
        super().__init__()
        self.bn = nn.BatchNorm1d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv1d(inplanes, outplanes, kernel_size=kernel_size, stride=stride, padding=kernel_size//2)

        self.bn2 = nn.BatchNorm1d(inplanes)
        self.conv2 = nn.Conv1d(inplanes, outplanes, kernel_size=1, stride=stride)

    def forward(self, x):
        identity = self.conv2(self.relu(self.bn2(x)))
        out = self.conv(self.relu(self.bn(x)))
        out += identity

        return out

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=[1, 2], stride=[1, 2])
        self.conv2 = nn.Conv2d(16, 16, kernel_size=[1, 2], stride=[1, 1])

        layers = [Block1(16, 16, [1,3], [0, 3//2]) for _ in range(3)]
        self.layers1 = nn.Sequential(*layers)

        layers = [Block1(16, 16, [1,5], [0, 5//2]) for _ in range(3)]
        self.layers2 = nn.Sequential(*layers)

        layers = [Block1(16, 16, [1,7], [0, 7//2]) for _ in range(3)]
        self.layers3 = nn.Sequential(*layers)

        layers = [Block2(kernel_size=3) for _ in range(2)]
        self.layers11 = nn.Sequential(*layers)

        layers = [Block2(kernel_size=5) for _ in range(2)]
        self.layers21 = nn.Sequential(*layers)

        layers = [Block2(kernel_size=7) for _ in range(2)]
        self.layers31 = nn.Sequential(*layers)

        self.fc = nn.Linear(960, 2)

    def forward(self, x):
        """ x.shape: batch_size x 20 x 125 (500 x 250hz = 125)
        """
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)

        batch = x.shape[0]
        length = x.shape[-1]

        x1 = self.layers1(x).reshape(batch, -1, length)
        x2 = self.layers2(x).reshape(batch, -1, length)
        x3 = self.layers3(x).reshape(batch, -1, length)

        x1 = self.layers11(x1)
        x2 = self.layers21(x2)
        x3 = self.layers31(x3)

        x1 = x1.mean(-1)
        x2 = x2.mean(-1)
        x3 = x3.mean(-1)
        #print(f"x1.shape:{x1.shape}")
        #print(f"x2.shape:{x2.shape}")
        #print(f"x3.shape:{x3.shape}")

        x = torch.cat([x1,x2,x3], dim=-1)
        x = self.fc(x).sigmoid()
        return x

if __name__ == "__main__":
    model = Model()
    x = torch.rand(1, 20, 125)
    x = model(x)
    print(f"x.shape: {x.shape}")
    print(x)





