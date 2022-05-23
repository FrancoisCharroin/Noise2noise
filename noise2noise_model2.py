import torch
import torch.nn as nn
import 'module.py' as mm

######################################## unet ######################################################

class UNet(nn.Module):
    """Custom U-Net architecture for Noise2Noise (see Appendix, Table 2)."""

    def __init__(self, in_channels=3, out_channels=3):
        """Initializes U-Net."""

        super(UNet, self).__init__()

        # Layers: enc_conv0, enc_conv1, pool1
        self._block1 = mm.Sequential(
            mm.Conv2d(in_channels, 48, 3, stride=1, padding=1),
            mm.LeakyReLU(0.1),
            mm.Conv2d(48, 48, 3, padding=1),
            mm.LeakyReLU(0.1),
            nn.MaxPool2d(2))

        # Layers: enc_conv(i), pool(i); i=2..5
        self._block2 = mm.Sequential(
            mm.Conv2d(48, 48, 3, stride=1, padding=1),
            mm.LeakyReLU(0.1),
            nn.MaxPool2d(2))

        # Layers: enc_conv6, upsample5
        self._block3 = mm.Sequential(
            mm.Conv2d(48, 48, 3, stride=1, padding=1),
            mm.LeakyReLU(0.1),
            mm.TransposeConv2d(48, 48, 3, stride=2, padding=1, output_padding=1))
            #nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_conv5a, dec_conv5b, upsample4
        self._block4 = mm.Sequential(
            mm.Conv2d(96, 96, 3, stride=1, padding=1),
            mm.LeakyReLU(0.1),
            mm.Conv2d(96, 96, 3, stride=1, padding=1),
            mm.LeakyReLU(0.1),
            mm.TransposeConv2d(96, 96, 3, stride=2, padding=1, output_padding=1))
            #nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_deconv(i)a, dec_deconv(i)b, upsample(i-1); i=4..2
        self._block5 = mm.Sequential(
            mm.Conv2d(144, 96, 3, stride=1, padding=1),
            mm.LeakyReLU(0.1),
            mm.Conv2d(96, 96, 3, stride=1, padding=1),
            mm.LeakyReLU(0.1),
            mm.TransposeConv2d(96, 96, 3, stride=2, padding=1, output_padding=1))
            #nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_conv1a, dec_conv1b, dec_conv1c,
        self._block6 = mm.Sequential(
            mm.Conv2d(96 + in_channels, 64, 3, stride=1, padding=1),
            mm.LeakyReLU(0.1),
            mm.Conv2d(64, 32, 3, stride=1, padding=1),
            mm.LeakyReLU(0.1),
            mm.Conv2d(32, out_channels, 3, stride=1, padding=1),
            nn.ReLU(inplace=True))

        # Initialize weights
        self._init_weights()


    def _init_weights(self):
        """Initializes weights using He et al. (2015)."""

        for m in self.modules():
            if isinstance(m, mo.TransposeConv2d) or isinstance(m, mo.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()


    def forward(self, x):
        """Through encoder, then decoder by adding U-skip connections. """

        # Encoder (3*32*32)
        pool1 = self._block1(x) # (48*16*16)
        pool2 = self._block2(pool1) # (48*8*8)
        pool3 = self._block2(pool2) # (48*4*4)
        pool4 = self._block2(pool3) # (48*2*2)
        pool5 = self._block2(pool4) # (48*1*1)

        # Decoder
        upsample5 = self._block3(pool5) # (48*2*2)
        concat5 = torch.cat((upsample5, pool4), dim=1) # (96*2*2)
        upsample4 = self._block4(concat5) # (96*4*4)
        concat4 = torch.cat((upsample4, pool3), dim=1) # (144*4*4)
        upsample3 = self._block5(concat4) # (96*8*8)
        concat3 = torch.cat((upsample3, pool2), dim=1) #(144*8*8)
        upsample2 = self._block5(concat3) # (96*16*16)
        concat2 = torch.cat((upsample2, pool1), dim=1) # (144*16*16)
        upsample1 = self._block5(concat2) # (96*32*32)
        concat1 = torch.cat((upsample1, x), dim=1) # (99*32*32)

        # Final activation
        return self._block6(concat1) # (3*32*32)
