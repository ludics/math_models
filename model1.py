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
    def __init__(self, inplanes=20, outplanes=20, kernel_size=3, stride=2):
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

        layers = [Block2(kernel_size=3) for _ in range(2)]
        self.layers11 = nn.Sequential(*layers)

        layers = [Block2(kernel_size=5) for _ in range(2)]
        self.layers21 = nn.Sequential(*layers)

        layers = [Block2(kernel_size=7) for _ in range(2)]
        self.layers31 = nn.Sequential(*layers)

        layers = [Block2(kernel_size=9) for _ in range(2)]
        self.layers31 = nn.Sequential(*layers)

        layers = [Block2(kernel_size=11) for _ in range(2)]
        self.layers31 = nn.Sequential(*layers)

        layers = [Block2(kernel_size=13) for _ in range(2)]
        self.layers31 = nn.Sequential(*layers)


        self.fc = nn.Linear(120, 2)

    def forward(self, x):
        """ x.shape: batch_size x 20 x 125 (500 x 250hz = 125)
        """
        batch = x.shape[0]
        length = x.shape[-1]


        x1 = self.layers11(x)
        x2 = self.layers21(x)
        x3 = self.layers31(x)
        x4 = self.layers31(x)
        x5 = self.layers31(x)
        x6 = self.layers31(x)

        #print(f"x1.shape:{x1.shape}")
        #print(f"x2.shape:{x2.shape}")
        #print(f"x3.shape:{x3.shape}")
        #print(f"x4.shape:{x3.shape}")

        x1 = x1.mean(-1)
        x2 = x2.mean(-1)
        x3 = x3.mean(-1)
        x4 = x4.mean(-1)
        x5 = x5.mean(-1)
        x6 = x6.mean(-1)


        x = torch.cat([x1,x2,x3, x4, x5, x6], dim=-1)
        #print(f"x.shape:{x.shape}")
        x = self.fc(x)
        return x

if __name__ == "__main__":
    model = Model()
    x = torch.rand(1, 20, 100)
    x = model(x)
    print(f"x.shape: {x.shape}")
    print(x)





