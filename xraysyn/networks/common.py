import torch
import torch.nn as nn
from .blocks import ConvolutionBlock, ResidualBlock, FullyConnectedBlock


class ResEncoder(nn.Module):
    def __init__(self, input_ch, base_ch, num_down, num_residual,
                 conv="con2d", pad='reflect2d', res_norm='instance2d', down_norm='instance2d'):
        super(ResEncoder, self).__init__()

        self.conv0 = ConvolutionBlock(
            in_channels=input_ch, out_channels=base_ch, kernel_size=7, stride=1,
            padding=3, conv=conv, pad=pad, norm=down_norm, activ='relu')

        output_ch = base_ch
        for i in range(1, num_down + 1):
            m = ConvolutionBlock(
                in_channels=output_ch, out_channels=output_ch * 2, kernel_size=4,
                stride=2, padding=1, conv=conv, pad=pad, norm=down_norm, activ='relu')
            setattr(self, "conv{}".format(i), m)
            output_ch *= 2

        for i in range(num_residual):
            setattr(self, "res{}".format(i),
                    ResidualBlock(output_ch, conv=conv, pad=pad, norm=res_norm, activ='relu'))

        self.layers = [getattr(self, "conv{}".format(i)) for i in range(num_down + 1)] + \
                      [getattr(self, "res{}".format(i)) for i in range(num_residual)]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ResDecoder(nn.Module):
    def __init__(self, output_ch, base_ch, num_up, num_residual,
                 conv="con2d", pad='reflect2d', res_norm='instance2d', up_norm='layer'):
        super(ResDecoder, self).__init__()
        input_ch = base_ch * 2 ** num_up
        input_chs = []

        for i in range(num_residual):
            setattr(self, "res{}".format(i),
                    ResidualBlock(input_ch, conv=conv, pad=pad, norm=res_norm, activ='lrelu'))
            input_chs.append(input_ch)

        for i in range(num_up):
            m = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                ConvolutionBlock(
                    in_channels=input_ch, out_channels=input_ch // 2, kernel_size=5,
                    stride=1, padding=2, conv=conv, pad=pad, norm=up_norm, activ='lrelu'))
            setattr(self, "conv{}".format(i), m)
            input_chs.append(input_ch)
            input_ch //= 2

        m = ConvolutionBlock(
            in_channels=base_ch, out_channels=output_ch, kernel_size=7,
            stride=1, padding=3, conv=conv, pad=pad, norm='none', activ='tanh')
        setattr(self, "conv{}".format(num_up), m)
        input_chs.append(base_ch)

        self.layers = [getattr(self, "res{}".format(i)) for i in range(num_residual)] + \
                      [getattr(self, "conv{}".format(i)) for i in range(num_up + 1)]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator

    This class is adopted from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer="batch",
                 dimension="2d"):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()

        conv, inst_norm, batch_norm = {
            "2d": (nn.Conv2d, nn.InstanceNorm2d, nn.BatchNorm2d),
            "3d": (nn.Conv3d, nn.InstanceNorm3d, nn.BatchNorm3d)}[dimension]

        if type(norm_layer) is str:
            norm_layer = {
                "layer": nn.LayerNorm, "instance": inst_norm,
                "batch": batch_norm, "none": None}[norm_layer]

        kw = 4
        padw = 1
        sequence = [conv(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        use_bias = norm_layer != batch_norm  # no need to use bias as BatchNorm2d has affine parameters
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                            conv(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw,
                                 bias=use_bias)] + \
                        ([norm_layer(ndf * nf_mult)] if norm_layer else []) + [nn.LeakyReLU(0.2, True)]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
                        conv(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw,
                             bias=use_bias)] + \
                    ([norm_layer(ndf * nf_mult)] if norm_layer else []) + [nn.LeakyReLU(0.2, True)]

        sequence += [conv(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, predictions, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            predictions (tensor list) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        all_losses = []
        for prediction in predictions:
            if self.gan_mode in ['lsgan', 'vanilla']:
                target_tensor = self.get_target_tensor(prediction, target_is_real)
                loss = self.loss(prediction, target_tensor)
            elif self.gan_mode == 'wgangp':
                if target_is_real:
                    loss = -prediction.mean()
                else:
                    loss = prediction.mean()
            all_losses.append(loss)

        return sum(all_losses)


def _calc_grad_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028
    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss
    Returns the gradient penalty loss
    """
    # print('DIMENSION',real_data.shape, fake_data.shape)
    if lambda_gp > 0.0:
        if type == 'real':  # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(
                *real_data.shape)
            alpha = alpha.to(device)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp  # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


def init_weights(net, init_type='normal', init_param=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_param)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_param)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orth':
                nn.init.orthogonal_(m.weight.data, gain=init_param)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, init_param)
            nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


if __name__ == "__main__":
    import torch
    import torch.nn as nn
    from xraysyn.networks.common import ResEncoder, ResDecoder

    image = torch.rand(8, 1, 128, 128, 128, requires_grad=True)
    encoder = ResEncoder(1, 2, 2, 2, conv="conv3d", pad='zero3d', res_norm='instance3d', down_norm='instance3d')
    decoder = ResDecoder(1, 2, 2, 2, conv="conv3d", pad='zero3d', res_norm='instance3d')

    image = image.cuda()
    encoder.cuda()
    decoder.cuda()

    feat = encoder(image)
    out = decoder(feat)
    loss = out.mean()
    loss.backward()