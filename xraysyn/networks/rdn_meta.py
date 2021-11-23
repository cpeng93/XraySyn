import torch
import torch.nn as nn

def make_model():
    return CT2XrayEst()


class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)


class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G = growRate
        C = nConvLayers

        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c * G, G))
        self.convs = nn.Sequential(*convs)

        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C * G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x



class GenWeights(nn.Module):
    def __init__(self):
        super(GenWeights,self).__init__()

        self.meta_block=nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, padding=1, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 3, padding=1, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1024, 3, padding=1, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 2048, 3, padding=1, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(2048, 4096, 3, padding=1, stride=2)
        )

    def forward(self,x):
        output = self.meta_block(x)
        return output


class CT2XrayEst(nn.Module):
    def __init__(self):
        super(CT2XrayEst, self).__init__()
        G0 = 64
        kSize = 3

        # number of RDB blocks D, conv layers within the blocks C, out channels G within the last layer of the blocks,
        self.D, C, G = {
            'A': (20, 6, 32),
            'B': (16, 8, 64),
            'C': (4, 8, 32),
        }['B']
        self.device = torch.device('cuda')
        # Shallow feature extraction net
        self.SFENet1 = nn.Conv2d(2, G0, kSize, padding=(kSize - 1) // 2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)

        # Redidual dense blocks and dense feature fusion
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(
                RDB(growRate0=G0, growRate=G, nConvLayers=C)
            )

        # Global Feature Fusion
        self.GFF = nn.Sequential(*[
            nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1),
            nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.ReLU()
        ])

        self.UPNet = nn.Sequential(*[
            nn.Conv2d(G0, G * 4, kSize, padding=(kSize-1)//2, stride=1),
            nn.PixelShuffle(2),
            nn.Conv2d(G, 2, kSize, padding=(kSize - 1) // 2, stride=1)
        ])

        self.meta = GenWeights()


    def norm(self, inp):
        inp = inp - inp.min()
        return inp/inp.max()

    def forward(self, x, ref):
        # print("network input size: ", x.size(), T_in.shape)
        f__1 = self.SFENet1(x)
        x = self.SFENet2(f__1)

        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)

        x = self.GFF(torch.cat(RDBs_out, 1))
        x += f__1


        weights = self.meta(ref).view(ref.size(0),64,64,1,1)
        inp = nn.functional.unfold(x, 1, padding=0)
        out = inp.transpose(1, 2).matmul(weights.view(weights.size(0), weights.size(1), -1).transpose(1,2)).transpose(1, 2)
        out = out.view(ref.shape[0], 64, 128, 128)
        out = self.UPNet(out)


        return out