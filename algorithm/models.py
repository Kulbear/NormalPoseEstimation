import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(EncoderBlock, self).__init__()
        self.conv_block = nn.Sequential(
            torch.nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, stride=2, bias=False),
            torch.nn.BatchNorm2d(out_channel),
            torch.nn.ReLU(),

            torch.nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, stride=1, bias=False),
            torch.nn.BatchNorm2d(out_channel),
            torch.nn.ReLU(),

            torch.nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, stride=1, bias=False),
            torch.nn.BatchNorm2d(out_channel),
            torch.nn.ReLU(),
        )

    def forward(self, _input):
        out = self.conv_block(_input)
        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channel, out_channel, is_last_layer=False):
        super(DecoderBlock, self).__init__()
        if is_last_layer:
            self.conv_block = nn.Sequential(
                torch.nn.ConvTranspose2d(in_channel, out_channel, kernel_size=3, padding=1,
                                         output_padding=1, stride=2, bias=False),
                torch.nn.BatchNorm2d(out_channel),
                torch.nn.ReLU(),

                torch.nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, stride=1, bias=False),
                torch.nn.BatchNorm2d(out_channel),
                torch.nn.ReLU(),

                torch.nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, stride=1, bias=False),
                torch.nn.BatchNorm2d(out_channel),
            )
        else:
            self.conv_block = nn.Sequential(
                torch.nn.ConvTranspose2d(in_channel, out_channel, kernel_size=3, padding=1,
                                         output_padding=1, stride=2, bias=False),
                torch.nn.BatchNorm2d(out_channel),
                torch.nn.ReLU(),

                torch.nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, stride=1, bias=False),
                torch.nn.BatchNorm2d(out_channel),
                torch.nn.ReLU(),

                torch.nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, stride=1, bias=False),
                torch.nn.BatchNorm2d(out_channel),
                torch.nn.ReLU(),
            )

    def forward(self, _input):
        out = self.conv_block(_input)
        return out


class Color2Normal(nn.Module):
    def __init__(self, temperature):
        super(Color2Normal, self).__init__()
        self.temperature = temperature
        # input_channel, out_channel1, out_channel2
        self.out_channel1 = 3
        self.out_channel2 = 2
        self.input_channel = 3

        self.encoder1_1 = EncoderBlock(self.input_channel, 16)
        self.encoder1_2 = EncoderBlock(16, 32)
        self.encoder1_3 = EncoderBlock(32, 64)
        self.encoder1_4 = EncoderBlock(64, 128)
        self.encoder1_5 = EncoderBlock(128, 256)
        self.encoder1_6 = EncoderBlock(256, 512)
        self.encoder1_7 = EncoderBlock(512, 1024)

        self.decoder1_1 = DecoderBlock(1024, 512)
        self.decoder1_2 = DecoderBlock(512, 256)
        self.decoder1_3 = DecoderBlock(256, 128)
        self.decoder1_4 = DecoderBlock(128, 64)
        self.decoder1_5 = DecoderBlock(64, 32)
        self.decoder1_6 = DecoderBlock(32, 16)
        self.decoder1_7 = DecoderBlock(16, self.out_channel1, True)

        self.decoder2_1 = DecoderBlock(1024, 512)
        self.decoder2_2 = DecoderBlock(512, 256)
        self.decoder2_3 = DecoderBlock(256, 128)
        self.decoder2_4 = DecoderBlock(128, 64)
        self.decoder2_5 = DecoderBlock(64, 32)
        self.decoder2_6 = DecoderBlock(32, 16)
        self.decoder2_7 = DecoderBlock(16, self.out_channel2, True)

    def forward(self, _input1):
        encoder1_1 = self.encoder1_1(_input1)
        encoder1_2 = self.encoder1_2(encoder1_1)
        encoder1_3 = self.encoder1_3(encoder1_2)
        encoder1_4 = self.encoder1_4(encoder1_3)
        encoder1_5 = self.encoder1_5(encoder1_4)
        encoder1_6 = self.encoder1_6(encoder1_5)
        encoder_out1 = self.encoder1_7(encoder1_6)

        decoder1_1 = self.decoder1_1(encoder_out1) + encoder1_6
        decoder1_2 = self.decoder1_2(decoder1_1) + encoder1_5
        decoder1_3 = self.decoder1_3(decoder1_2) + encoder1_4
        decoder1_4 = self.decoder1_4(decoder1_3) + encoder1_3
        decoder1_5 = self.decoder1_5(decoder1_4) + encoder1_2
        decoder1_6 = self.decoder1_6(decoder1_5) + encoder1_1
        pred_normal = self.decoder1_7(decoder1_6)

        decoder2_1 = self.decoder2_1(encoder_out1) + encoder1_6
        decoder2_2 = self.decoder2_2(decoder2_1) + encoder1_5
        decoder2_3 = self.decoder2_3(decoder2_2) + encoder1_4
        decoder2_4 = self.decoder2_4(decoder2_3) + encoder1_3
        decoder2_5 = self.decoder2_5(decoder2_4) + encoder1_2
        decoder2_6 = self.decoder2_6(decoder2_5) + encoder1_1
        category_mask = self.decoder2_7(decoder2_6)
        out_category_mask = category_mask * self.temperature

        pred_mask = (torch.argmax(category_mask, dim=1, keepdim=True) > 0).float()  # [N, 1, H, W]
        return out_category_mask, pred_normal, pred_mask

