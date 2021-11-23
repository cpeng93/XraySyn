import torch
import torch.nn as nn


class UnetGenerator(nn.Module):
    def __init__(
        self, input_nc, output_nc, dimension="2d", mask_nc=0, num_downs=5, ngf=64,
        norm_layer="none", up_layer="upsample2D", partial_conv=False, use_dropout=False,
        use_tanh=True, output_feats=False):
        assert num_downs >= 5
        super(UnetGenerator, self).__init__()

        norm_layer = {
            "batch": {"2d": nn.BatchNorm2d, "3d": nn.BatchNorm3d}[dimension],
            "instance": {"2d": nn.InstanceNorm2d, "3d": nn.InstanceNorm3d}[dimension],
            "none": None}[norm_layer]

        self.down0 = UnetDown(input_nc, ngf, mask_nc, dimension=dimension,
            norm_layer=None, partial_conv=partial_conv)
        self.down1 = UnetDown(ngf, ngf * 2, mask_nc, dimension=dimension,
            norm_layer=norm_layer, partial_conv=partial_conv)
        self.down2 = UnetDown(ngf * 2, ngf * 4, mask_nc, dimension=dimension,
            norm_layer=norm_layer, partial_conv=partial_conv)
        self.down3 = UnetDown(ngf * 4, ngf * 8, mask_nc, dimension=dimension,
            norm_layer=norm_layer, partial_conv=partial_conv)
        for i in range(4, num_downs):
            setattr(
                self, "down{}".format(i),
                UnetDown(ngf * 8, ngf * 8, mask_nc, dimension=dimension,
                    norm_layer=norm_layer, partial_conv=partial_conv))

        setattr(
            self, "up{}".format(num_downs - 1),
            UnetUp(ngf * 8, ngf * 8, mask_nc, dimension=dimension,
                norm_layer=norm_layer, up_layer=up_layer, partial_conv=partial_conv))
        for i in range(num_downs - 2, 3, -1):
            setattr(
                self, "up{}".format(i),
                UnetUp(ngf * 16, ngf * 8, mask_nc, dimension=dimension,
                    use_dropout=use_dropout, norm_layer=norm_layer, up_layer=up_layer, partial_conv=partial_conv))
        self.up3 = UnetUp(ngf * 16, ngf * 4, mask_nc, dimension=dimension,
            norm_layer=norm_layer, up_layer=up_layer, partial_conv=partial_conv)
        self.up2 = UnetUp(ngf * 8, ngf * 2, mask_nc, dimension=dimension,
            norm_layer=norm_layer, up_layer=up_layer, partial_conv=partial_conv)
        self.up1 = UnetUp(ngf * 4, ngf, mask_nc, dimension=dimension,
            norm_layer=norm_layer, up_layer=up_layer, partial_conv=partial_conv)
        self.up0 = UnetUp(ngf * 2, output_nc, mask_nc, dimension=dimension,
            up_layer=up_layer, final=True, partial_conv=partial_conv, use_tanh=use_tanh)
        self.num_downs = num_downs
        self.output_feats = output_feats

    def forward(self, x):
        x0_down, x1_down = [None], [x]
        for i in range(self.num_downs):
            down = getattr(self, "down{}".format(i))
            x0, x1 = down(x1_down[-1])
            x0_down.append(x0)
            x1_down.append(x1)

        y_up = x1_down[-1]
        if self.output_feats:
            feats = [y_up]
            for i in range(self.num_downs):
                up = getattr(self, "up{}".format(self.num_downs - 1 - i))
                y_up = up(y_up, x0_down[-2 - i])
                feats.append(y_up)
            return y_up, feats
        else:
            for i in range(self.num_downs):
                up = getattr(self, "up{}".format(self.num_downs - 1 - i))
                y_up = up(y_up, x0_down[-2 - i])
            return y_up


class UnetEncoder(nn.Module):
    def __init__(
        self, input_nc, dimension="2d", mask_nc=0, num_downs=5, ngf=64,
        norm_layer="none", partial_conv=False):
        assert num_downs >= 5
        super(UnetEncoder, self).__init__()

        norm_layer = {
            "batch": {"2d": nn.BatchNorm2d, "3d": nn.BatchNorm3d}[dimension],
            "instance": {"2d": nn.InstanceNorm2d, "3d": nn.InstanceNorm3d}[dimension],
            "none": None}[norm_layer]

        self.down0 = UnetDown(input_nc, ngf, mask_nc, dimension=dimension,
            norm_layer=None, partial_conv=partial_conv)
        self.down1 = UnetDown(ngf, ngf * 2, mask_nc, dimension=dimension,
            norm_layer=norm_layer, partial_conv=partial_conv)
        self.down2 = UnetDown(ngf * 2, ngf * 4, mask_nc, dimension=dimension,
            norm_layer=norm_layer, partial_conv=partial_conv)
        self.down3 = UnetDown(ngf * 4, ngf * 8, mask_nc, dimension=dimension,
            norm_layer=norm_layer, partial_conv=partial_conv)
        for i in range(4, num_downs):
            setattr(
                self, "down{}".format(i),
                UnetDown(ngf * 8, ngf * 8, mask_nc, dimension=dimension,
                    norm_layer=norm_layer, partial_conv=partial_conv))

        self.num_downs = num_downs

    def forward(self, x):
        sides, y = [], x
        for i in range(self.num_downs):
            down = getattr(self, "down{}".format(i))
            side, y = down(y)
            sides.append(side)

        return y, sides


class UnetNewEncoder(nn.Module):
    def __init__(
        self, input_nc, dimension="2d", mask_nc=0, num_downs=5, ngf=64,
        norm_layer="none", partial_conv=False):
        assert num_downs >= 5
        super(UnetNewEncoder, self).__init__()

        norm_layer = {
            "batch": {"2d": nn.BatchNorm2d, "3d": nn.BatchNorm3d}[dimension],
            "instance": {"2d": nn.InstanceNorm2d, "3d": nn.InstanceNorm3d}[dimension],
            "none": None}[norm_layer]

        self.down0 = UnetDown(input_nc, ngf, mask_nc, dimension=dimension,
            norm_layer=None, partial_conv=partial_conv)
        self.down1 = UnetDown(ngf, ngf * 2, mask_nc, dimension=dimension,
            norm_layer=norm_layer, partial_conv=partial_conv)
        self.down2 = UnetDown(ngf * 2, ngf * 4, mask_nc, dimension=dimension,
            norm_layer=norm_layer, partial_conv=partial_conv)
        self.down3 = UnetDown(ngf * 4, ngf * 8, mask_nc, dimension=dimension,
            norm_layer=norm_layer, partial_conv=partial_conv)
        self.down4 = UnetDown(ngf * 8, ngf * 8, mask_nc, dimension=dimension,
            norm_layer=norm_layer, partial_conv=partial_conv)
        self.down5 = UnetNewDown(ngf * 8, ngf * 16, mask_nc, dimension=dimension,
            norm_layer=norm_layer, partial_conv=partial_conv)
        self.down6 = UnetNewDown(ngf * 16, ngf * 16, mask_nc, dimension=dimension,
            norm_layer=norm_layer, partial_conv=partial_conv)
        self.down7 = UnetNewDown(ngf * 16, ngf * 32, mask_nc, dimension=dimension,
            norm_layer=norm_layer, partial_conv=partial_conv)
        self.down8 = UnetNewDown(ngf * 32, ngf * 16, mask_nc, dimension=dimension,
            norm_layer=norm_layer, partial_conv=partial_conv)
        self.down9 = UnetNewDown(ngf * 16, ngf * 8, mask_nc, dimension=dimension,
            norm_layer=norm_layer, partial_conv=partial_conv)
        self.down10 = UnetNewDown(ngf * 8, ngf * 4, mask_nc, dimension=dimension,
            norm_layer=norm_layer, partial_conv=partial_conv)
        self.down11 = UnetNewDown(ngf * 4, ngf * 2, mask_nc, dimension=dimension,
            norm_layer=norm_layer, partial_conv=partial_conv)
        self.down12 = UnetNewDown(ngf * 2, 32, mask_nc, dimension=dimension,
            norm_layer=norm_layer, partial_conv=partial_conv)
        self.down13 = nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1)
        self.num_downs = 13

    def forward(self, x):
        for i in range(self.num_downs):
            down = getattr(self, "down{}".format(i))
            _, x = down(x)

        x = self.down13(x)

        return x


class UnetDecoder(nn.Module):
    def __init__(
        self, output_nc, dimension="2d", mask_nc=0, num_ups=5, ngf=64,
        norm_layer="none", num_inputs=1, up_layer="upsample", partial_conv=False,
        use_dropout=False, use_tanh=True):
        assert num_ups >= 5
        super(UnetDecoder, self).__init__()

        norm_layer = {
            "batch": {"2d": nn.BatchNorm2d, "3d": nn.BatchNorm3d}[dimension],
            "instance": {"2d": nn.InstanceNorm2d, "3d": nn.InstanceNorm3d}[dimension],
            "none": None}[norm_layer]
        setattr(
            self, "up{}".format(num_ups - 1),
            UnetUp(ngf * num_inputs * 8, ngf * 8, mask_nc, dimension=dimension,
                norm_layer=norm_layer, up_layer=up_layer, partial_conv=partial_conv))


        for i in range(num_ups - 2, 3, -1):
            setattr(
                self, "up{}".format(i),
                UnetUp(ngf * (num_inputs + 1) * 8, ngf * 8, mask_nc, dimension=dimension,
                    use_dropout=use_dropout, norm_layer=norm_layer, up_layer=up_layer, partial_conv=partial_conv))
        self.up3 = UnetUp(ngf * (num_inputs + 1) * 8, ngf * 4, mask_nc, dimension=dimension,
            norm_layer=norm_layer, up_layer=up_layer, partial_conv=partial_conv)
        self.up2 = UnetUp(ngf * (num_inputs + 1) * 4, ngf * 2, mask_nc, dimension=dimension,
            norm_layer=norm_layer, up_layer=up_layer, partial_conv=partial_conv)
        self.up1 = UnetUp(ngf * (num_inputs + 1) * 2, ngf, mask_nc, dimension=dimension,
            norm_layer=norm_layer, up_layer=up_layer, partial_conv=partial_conv)
        self.up0 = UnetUp(ngf * (num_inputs + 1), output_nc, mask_nc, dimension=dimension,
            up_layer=up_layer, final=True, partial_conv=partial_conv, use_tanh=use_tanh)
        self.num_ups = num_ups

    def forward(self, x, sides):
        y_up = x
        for i in range(self.num_ups-1):
            up = getattr(self, "up{}".format(self.num_ups - 1 - i))
            y_up = up(y_up, sides[-2 - i])
        y_up = self.up0(y_up)
        return y_up


class UnetDown(nn.Module):
    def __init__(
        self, input_nc, output_nc, mask_nc=1, dimension="2d",
        norm_layer=nn.BatchNorm2d, partial_conv=False
    ):
        super(UnetDown, self).__init__()

        conv_layer = {"2d": nn.Conv2d, "3d": nn.Conv3d}[dimension]
        self.conv = nn.utils.spectral_norm(conv_layer(
            input_nc + mask_nc, output_nc, kernel_size=3, stride=2, padding=1))
        if norm_layer is not None:
            self.norm = norm_layer(output_nc, affine=True)
        self.mask_nc = mask_nc
        self.leaky_relu = nn.LeakyReLU(0.2, True)
        self.partial_conv = partial_conv

    def forward(self, x):
        if self.mask_nc > 0:
            if self.partial_conv:
                x, y = (
                    x[:, :-self.mask_nc, ...],
                    x[:, -self.mask_nc:, ...]
                )
                x = x * (1 - y)
            else:
                y = x[:, -self.mask_nc:, ...]

        if hasattr(self, "norm"): x0 = self.norm(self.conv(x))
        else: x0 = self.conv(x)
        x1 = self.leaky_relu(x0)

        if self.mask_nc == 0: return x0, x1
        else: return torch.cat([x0, y], 1), torch.cat([x1, y], 1)


class UnetNewDown(nn.Module):
    def __init__(
        self, input_nc, output_nc, mask_nc=1, dimension="2d",
        norm_layer=nn.BatchNorm2d, partial_conv=False
    ):
        super(UnetNewDown, self).__init__()

        conv_layer = {"2d": nn.Conv2d, "3d": nn.Conv3d}[dimension]
        self.conv = nn.utils.spectral_norm(conv_layer(
            input_nc + mask_nc, output_nc, kernel_size=3, stride=1, padding=1))
        if norm_layer is not None:
            self.norm = norm_layer(output_nc, affine=True)
        self.mask_nc = mask_nc
        self.leaky_relu = nn.LeakyReLU(0.2, True)
        self.partial_conv = partial_conv

    def forward(self, x):
        if self.mask_nc > 0:
            if self.partial_conv:
                x, y = (
                    x[:, :-self.mask_nc, ...],
                    x[:, -self.mask_nc:, ...]
                )
                x = x * (1 - y)
            else:
                y = x[:, -self.mask_nc:, ...]

        if hasattr(self, "norm"): x0 = self.norm(self.conv(x))
        else: x0 = self.conv(x)
        x1 = self.leaky_relu(x0)

        if self.mask_nc == 0: return x0, x1
        else: return torch.cat([x0, y], 1), torch.cat([x1, y], 1)


class UnetUp(nn.Module):
    def __init__(
        self, input_nc, output_nc, mask_nc=1, dimension="2d", final=False, use_dropout=False,
        norm_layer=nn.BatchNorm2d, up_layer="upsample2D", partial_conv=False, use_tanh=True):
        super(UnetUp, self).__init__()
        # print('output_nc: ', input_nc + mask_nc, output_nc)
        conv_layer = {"2d": nn.Conv2d, "3d": nn.Conv3d}[dimension]
        deconv_layer = {"2d": nn.ConvTranspose2d, "3d": nn.ConvTranspose3d}[dimension]
        # print(up_layer)
        self.deconv = {
            "deconv": nn.utils.spectral_norm(deconv_layer(
                input_nc + mask_nc, output_nc, kernel_size=3,
                stride=2, padding=1)),
            "upsample2D":  nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.utils.spectral_norm(conv_layer(
                    input_nc + mask_nc, output_nc, kernel_size=3,
                    stride=1, padding=1))),
            "upsample3D":  nn.Sequential(
                nn.Upsample(scale_factor=2, mode='trilinear'),
                nn.utils.spectral_norm(conv_layer(
                    input_nc + mask_nc, output_nc, kernel_size=3,
                    stride=1, padding=1)))
        }[up_layer]

        if final:
            self.tanh = nn.Tanh() if use_tanh else nn.Identity()
        else:
            if norm_layer is not None:
                self.norm = norm_layer(output_nc, affine=True)
            if use_dropout: self.dropout = nn.Dropout(0.5)
            self.relu = nn.ReLU(True)
        self.partial_conv = partial_conv
        self.mask_nc = mask_nc

    def forward(self, x1, x2=None):
        if self.partial_conv and self.mask_nc > 0: x1 = x1[:, :-self.mask_nc, ...]
        y = self.deconv(x1)
        if hasattr(self, "tanh"):
            y = self.tanh(y)
        else:
            if hasattr(self, "norm"): y = self.norm(y)
            if hasattr(self, "dropout"): y = self.dropout(y)
            y = torch.cat([y, x2], 1)
            y = self.relu(y)
        return y
