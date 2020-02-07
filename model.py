import torch
import torch.nn as nn
import torch.nn.functional as F


class Inception_A(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception_A, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(True),
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_planes, n3x3red, kernel_size=1),
            nn.BatchNorm2d(n3x3red),
            nn.ReLU(True),
            nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(True),
        )

        # 1x1 conv -> 3x3 conv -> 3x3 conv branch
        self.b3 = nn.Sequential(
            nn.Conv2d(in_planes, n5x5red, kernel_size=1),
            nn.BatchNorm2d(n5x5red),
            nn.ReLU(True),
            nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
            nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1,y2,y3,y4], 1)

class Inception_B(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n7x7red, n7x7, pool_planes):
        super(Inception_B,self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(True),
        )

        # 1x1 conv -> 1xn conv -> nx1 conv branch (n == 7)
        self.b2 = nn.Sequential(
            nn.Conv2d(in_planes, n3x3red, kernel_size=1),
            nn.BatchNorm2d(n3x3red),
            nn.ReLU(True),
            nn.Conv2d(n3x3red, n3x3, kernel_size=(7, 1), padding=(3, 0)),
            nn.Conv2d(n3x3, n3x3, kernel_size=(1, 7), padding=(0, 3)),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(True),
        )

        # 1x1 conv -> 1x7 conv branch -> 7x1 conv -> 1x7 conv -> 7x1 conv branch
        self.b3 = nn.Sequential(
            nn.Conv2d(in_planes, n7x7red, kernel_size=1),
            nn.BatchNorm2d(n7x7red),
            nn.ReLU(True),
            nn.Conv2d(n7x7red, n7x7, kernel_size=(7, 1), padding=(3, 0)),
            nn.Conv2d(n7x7, n7x7, kernel_size=(1, 7), padding=(0, 3)),
            nn.BatchNorm2d(n7x7),
            nn.ReLU(True),

            nn.Conv2d(n7x7, n7x7, kernel_size=(7, 1), padding=(3, 0)),
            nn.Conv2d(n7x7, n7x7, kernel_size=(1, 7), padding=(0, 3)),
            nn.BatchNorm2d(n7x7),
            nn.ReLU(True),
        )
        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1,y2,y3,y4], 1)


class Inception_C(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception_C,self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(True),
        )
        # 1x1 conv -> 1x3 conv branch
        self.b2_1 = nn.Sequential(
            nn.Conv2d(in_planes, n3x3red, kernel_size=1),
            nn.BatchNorm2d(n3x3red),
            nn.ReLU(True),
            nn.Conv2d(n3x3red, n3x3, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(True),
        )

        # 1x1 conv -> 3x1 conv branch
        self.b2_2 = nn.Sequential(
            nn.Conv2d(in_planes, n3x3red, kernel_size=1),
            nn.BatchNorm2d(n3x3red),
            nn.ReLU(True),
            nn.Conv2d(n3x3red, n3x3, kernel_size=(3,1), padding=(1,0)),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(True),
        )

        # 1x1 conv -> 3x3 conv -> 1x3 conv branch
        self.b3_1 = nn.Sequential(
            nn.Conv2d(in_planes, n5x5red, kernel_size=1),
            nn.BatchNorm2d(n5x5red),
            nn.ReLU(True),
            nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
            nn.Conv2d(n5x5, n5x5, kernel_size=(1,3), padding=(0,1)),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
        )

        # 1x1 conv -> 3x3 conv -> 3x1 conv branch
        self.b3_2 = nn.Sequential(
            nn.Conv2d(in_planes, n5x5red, kernel_size=1),
            nn.BatchNorm2d(n5x5red),
            nn.ReLU(True),
            nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
            nn.Conv2d(n5x5, n5x5, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2_1(x)
        y3 = self.b2_2(x)
        y4 = self.b3_1(x)
        y5 = self.b3_2(x)
        y6 = self.b4(x)

        return torch.cat([y1,y2,y3,y4,y5,y6], 1)

class GoogLeNet(nn.Module):
    def __init__(self):
        super(GoogLeNet, self).__init__()
        # input 32 x 32 x 3
        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )
        # size 32 x 32 x 192
        self.a3 = Inception_A(192,  64,  96, 128, 16, 32, 32)
        self.b3 = Inception_A(256, 128, 128, 192, 32, 96, 64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        # size 128, 480, 16, 16
        self.a4 = Inception_B(480, 192,  96, 208, 48,  48,  64)
        self.b4 = Inception_B(512, 160, 112, 224, 24,  64,  64)
        self.c4 = Inception_B(512, 128, 128, 256, 24,  64,  64)
        self.d4 = Inception_B(512, 112, 144, 288, 32,  64,  64)
        self.e4 = Inception_B(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception_C(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception_C(1280, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.linear = nn.Linear(1536, 100)

    def forward(self, x):
        #print(x.size()) // [128, 3, 32, 32]
        out = self.pre_layers(x)
        # print(out.size()) //[128, 192, 32, 32]
        out = self.a3(out)
        #print(out.size()) // [128, 256, 32, 32]
        out = self.b3(out)
        #print(out.size()) // [128, 480, 32, 32]
        out = self.maxpool(out)
        out = self.a4(out)
        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)
        out = self.e4(out)
        out = self.maxpool(out)
        #print(out.size()) # torch.Size([128, 832, 8, 8])
        out = self.a5(out)
        # print(out.size()) #torch.Size([128, 1280, 8, 8])

        out = self.b5(out)
        # print(out.size()) #torch.Size([128, 1536, 8, 8])

        out = self.avgpool(out)
        # print(out.size()) #torch.Size([128, 1536, 1, 1])

        out = out.view(out.size(0), -1)
        # print(out.size()) #torch.Size([128, 1536])

        out = self.linear(out)
        return out