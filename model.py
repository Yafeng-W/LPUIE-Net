import torch
import torch.nn as nn
from torchvision.models.vgg import vgg19
from torch.distributions import kl
from utils import ResBlock, ConvBlock, Up, Compute_z, ConvBlock1
import torch.nn.functional as F


class CALayer1(nn.Module):
    """Channel attention(CA)"""
    def __init__(self, channel):
        super(CALayer1, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.CA = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.CA(y)
        return x * y


class PALayer1(nn.Module):
    """Pixel attention(PA)"""
    def __init__(self, channel):
        super(PALayer1, self).__init__()
        self.PA = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 3, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.InstanceNorm2d(64),
            nn.Conv2d(channel // 8, 1, 3, padding=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.PA(x)
        return x * y


class Block1(nn.Module):
    """parallel attention module(PAM)"""
    def __init__(self, ch):
        super(Block1, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(ch, 64, 3, padding=(3 // 2), bias=True)
        self.conv2 = nn.Conv2d(128, 64, 3, padding=(3 // 2), bias=True)
        self.calayer = CALayer1(64)
        self.palayer = PALayer1(64)
        self.conv3 = nn.Conv2d(ch, 64, 1)

    def forward(self, x):
        res = self.conv1(x)
        res1 = self.calayer(res)
        res2 = self.palayer(res)
        res3 = torch.cat((res1, res2), 1)
        res4 = self.conv2(res3)
        x1 = self.conv3(x)
        res5 = res4 + x1
        return res5


class Block2(nn.Module):
    def __init__(self, ch):
        super(Block2, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(ch, 128, 3, padding=(3 // 2), bias=True)
        self.conv2 = nn.Conv2d(256, 128, 3, padding=(3 // 2), bias=True)
        self.calayer = CALayer1(128)
        self.palayer = PALayer1(128)
        self.conv3 = nn.Conv2d(ch, 128, 1)

    def forward(self, x):
        res = self.conv1(x)
        res1 = self.calayer(res)
        res2 = self.palayer(res)
        res3 = torch.cat((res1, res2), 1)
        res4 = self.conv2(res3)
        x1 = self.conv3(x)
        res5 = res4 + x1
        return res5


class Block3(nn.Module):
    def __init__(self, ch):
        super(Block3, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(ch, 256, 3, padding=(3 // 2), bias=True)
        self.conv2 = nn.Conv2d(512, 256, 3, padding=(3 // 2), bias=True)
        self.calayer = CALayer1(256)
        self.palayer = PALayer1(256)
        self.conv3 = nn.Conv2d(ch, 256, 1)

    def forward(self, x):
        res = self.conv1(x)
        res1 = self.calayer(res)
        res2 = self.palayer(res)
        res3 = torch.cat((res1, res2), 1)
        res4 = self.conv2(res3)
        x1 = self.conv3(x)
        res5 = res4 + x1
        return res5


class Encoder1(nn.Module):
    def __init__(self, ch):
        super(Encoder1, self).__init__()
        self.relu_1 = nn.ReLU(inplace=True)
        self.block1 = Block1(ch)

        self.conv_e1 = nn.Conv2d(ch, 64, kernel_size=3, stride=1, padding=1)

        self.relu_e1 = nn.ReLU(inplace=True)
        self.tanh_e1 = nn.Tanh()
        self.conv_e101 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.conv_e102 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.conv_e103 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.conv_101 = nn.Conv2d(64 + 3, 64, kernel_size=3, stride=1, padding=1)
        self.upsample_e1 = F.upsample_nearest

    def forward(self, x):
        dehaze_1 = x
        dehaze_1 = self.relu_1(self.conv_e1(dehaze_1))
        dehaze_11 = self.block1(x)
        shape_out_1 = dehaze_1.data.size()
        shape_out_1 = shape_out_1[2:4]

        x101 = F.avg_pool2d(dehaze_1, 128)
        x102 = F.avg_pool2d(dehaze_1, 64)
        x103 = F.avg_pool2d(dehaze_1, 32)

        x1010 = self.upsample_e1(self.relu_e1(self.conv_e101(x101)), size=shape_out_1)
        x1020 = self.upsample_e1(self.relu_e1(self.conv_e102(x102)), size=shape_out_1)
        x1030 = self.upsample_e1(self.relu_e1(self.conv_e103(x103)), size=shape_out_1)
        dehaze_1 = torch.cat((x1010, x1020, x1030, dehaze_11), 1)
        dehaze_e1 = self.tanh_e1(self.conv_101(dehaze_1))
        return dehaze_e1


class Encoder2(nn.Module):
    def __init__(self, ch):
        super(Encoder2, self).__init__()
        self.relu_2 = nn.ReLU(inplace=True)
        self.block2 = Block2(ch)

        self.conv_e2 = nn.Conv2d(ch, 128, kernel_size=5, stride=1, padding=2)

        self.relu_e2 = nn.ReLU(inplace=True)
        self.tanh_e2 = nn.Tanh()
        self.conv_e201 = nn.Conv2d(128, 1, kernel_size=5, stride=1, padding=2)
        self.conv_e202 = nn.Conv2d(128, 1, kernel_size=5, stride=1, padding=2)
        self.conv_e203 = nn.Conv2d(128, 1, kernel_size=5, stride=1, padding=2)
        self.conv_201 = nn.Conv2d(128 + 3, 128, kernel_size=5, stride=1, padding=2)
        self.upsample_e2 = F.upsample_nearest

    def forward(self, x):
        dehaze_2 = x
        dehaze_2 = self.relu_2(self.conv_e2(dehaze_2))
        dehaze_21 = self.block2(x)
        shape_out_2 = dehaze_2.data.size()
        shape_out_2 = shape_out_2[2:4]

        x201 = F.avg_pool2d(dehaze_2, 128)
        x202 = F.avg_pool2d(dehaze_2, 64)
        x203 = F.avg_pool2d(dehaze_2, 32)

        x2010 = self.upsample_e2(self.relu_e2(self.conv_e201(x201)), size=shape_out_2)
        x2020 = self.upsample_e2(self.relu_e2(self.conv_e202(x202)), size=shape_out_2)
        x2030 = self.upsample_e2(self.relu_e2(self.conv_e203(x203)), size=shape_out_2)
        dehaze_2 = torch.cat((x2010, x2020, x2030, dehaze_21), 1)
        dehaze_e2 = self.tanh_e2(self.conv_201(dehaze_2))
        return dehaze_e2


class Encoder3(nn.Module):
    def __init__(self, ch):
        super(Encoder3, self).__init__()
        self.relu_3 = nn.ReLU(inplace=True)
        self.block3 = Block3(ch)

        self.conv_e3 = nn.Conv2d(ch, 256, kernel_size=7, stride=1, padding=3)

        self.relu_e3 = nn.ReLU(inplace=True)
        self.tanh_e3 = nn.Tanh()
        self.conv_e301 = nn.Conv2d(256, 1, kernel_size=7, stride=1, padding=3)
        self.conv_e302 = nn.Conv2d(256, 1, kernel_size=7, stride=1, padding=3)
        self.conv_e303 = nn.Conv2d(256, 1, kernel_size=7, stride=1, padding=3)
        self.conv_301 = nn.Conv2d(256 + 3, 256, kernel_size=7, stride=1, padding=3)
        self.upsample_e3 = F.upsample_nearest

    def forward(self, x):
        dehaze_3 = x
        dehaze_3 = self.relu_3(self.conv_e3(dehaze_3))
        dehaze_31 = self.block3(x)
        shape_out_3 = dehaze_3.data.size()
        shape_out_3 = shape_out_3[2:4]

        x301 = F.avg_pool2d(dehaze_3, 128)
        x302 = F.avg_pool2d(dehaze_3, 64)
        x303 = F.avg_pool2d(dehaze_3, 32)

        x3010 = self.upsample_e3(self.relu_e3(self.conv_e301(x301)), size=shape_out_3)
        x3020 = self.upsample_e3(self.relu_e3(self.conv_e302(x302)), size=shape_out_3)
        x3030 = self.upsample_e3(self.relu_e3(self.conv_e303(x303)), size=shape_out_3)
        dehaze_3 = torch.cat((x3010, x3020, x3030,  dehaze_31), 1)
        dehaze_e3 = self.tanh_e3(self.conv_301(dehaze_3))

        return dehaze_e3


class Decoder(nn.Module):
    def __init__(self, device):
        super(Decoder, self).__init__()
        self.device = device
        self.pr_encoder1 = Encoder1(3)
        self.po_encoder1 = Encoder1(6)
        self.pr_encoder2 = Encoder2(3)
        self.po_encoder2 = Encoder2(6)
        self.pr_encoder3 = Encoder3(3)
        self.po_encoder3 = Encoder3(6)

        self.pr_block = Block3(3)
        self.po_block = Block3(6)

        self.pr_conv = ResBlock(512)
        self.pr_conv_1 = ConvBlock1(ch_in=512, ch_out=256)
        self.pr_Up3 = Up()
        self.pr_UpConv3 = ConvBlock(ch_in=256 * 2, ch_out=256)
        self.pr_UpConv3_1 = ConvBlock1(ch_in=256, ch_out=128)
        self.pr_Up2 = Up()
        self.pr_UpConv2 = ConvBlock(ch_in=128 * 2, ch_out=128)
        self.pr_UpConv2_1 = ConvBlock1(ch_in=128, ch_out=64)
        self.pr_Up1 = Up()

        self.pr_UpConv1 = ConvBlock(ch_in=128, ch_out=64)

        self.po_conv = ResBlock(512)
        self.po_conv_1 = ConvBlock(ch_in=512, ch_out=256)
        self.po_Up3 = Up()
        self.po_UpConv3 = ConvBlock(ch_in=256 * 2, ch_out=256)
        self.po_UpConv3_1 = ConvBlock1(ch_in=256, ch_out=128)
        self.po_Up2 = Up()
        self.po_UpConv2 = ConvBlock(ch_in=128 * 2, ch_out=128)
        self.po_UpConv2_1 = ConvBlock1(ch_in=128, ch_out=64)
        self.pr_Up4 = Up()

        out_conv = []
        out_conv.append(ResBlock(64))
        out_conv.append(ResBlock(64))
        out_conv.append(nn.Conv2d(64, 3, kernel_size=1, padding=0))
        self.out_conv = nn.Sequential(*out_conv)
        z = 20
        self.compute_z_pr = Compute_z(z)
        self.compute_z_po = Compute_z(z)
        self.conv_u = nn.Conv2d(z, 128, kernel_size=1, padding=0)
        self.conv_s = nn.Conv2d(z, 128, kernel_size=1, padding=0)
        self.insnorm = nn.InstanceNorm2d(128)
        self.sigmoid = nn.Sigmoid()
        self.L_sum = 0.0
        self.L_count = 0
        self.L_avg = 0.0

    def get_L_avg(self):
        return self.L_sum / self.L_count if self.L_count > 0 else 0.0

    def reset_L(self):
        self.L_avg = self.get_L_avg()
        self.L_sum = 0.0
        self.L_count = 0

    def forward(self, Input, Target, training=True):
        pr_x1 = self.pr_encoder1.forward(Input)
        pr_x2 = self.pr_encoder2.forward(Input)
        pr_x3 = self.pr_encoder3.forward(Input)

        if training:
            po_x1 = self.po_encoder1.forward(torch.cat((Input, Target), dim=1))
            po_x2 = self.po_encoder2.forward(torch.cat((Input, Target), dim=1))
            po_x3 = self.po_encoder3.forward(torch.cat((Input, Target), dim=1))
            pr_d3 = self.pr_UpConv3_1(pr_x3)
            pr_d2 = pr_d3
            po_d3 = self.po_UpConv3_1(po_x3)
            po_d2 = po_d3
            pr_d2 = torch.cat((pr_x2, pr_d2), dim=1)
            pr_d2 = self.pr_UpConv2(pr_d2)
            pr_d2 = self.pr_UpConv2_1(pr_d2)
            pr_d1 = pr_d2
            po_d2 = torch.cat((po_x2, po_d2), dim=1)
            po_d2 = self.po_UpConv2(po_d2)
            po_d2 = self.po_UpConv2_1(po_d2)
            po_d1 = po_d2
            pr_d1 = torch.cat((pr_x1, pr_d1), dim=1)
            po_d1 = torch.cat((po_x1, po_d1), dim=1)
            pr_u_dist, pr_s_dist, _, _, _, _ = self.compute_z_pr(pr_d1)
            po_u_dist, po_s_dist, _, _, _, _ = self.compute_z_po(po_d1)
            po_latent_u = po_u_dist.rsample()
            po_latent_s = po_s_dist.rsample()
            po_latent_u = torch.unsqueeze(po_latent_u, -1)
            po_latent_u = torch.unsqueeze(po_latent_u, -1)
            po_latent_s = torch.unsqueeze(po_latent_s, -1)
            po_latent_s = torch.unsqueeze(po_latent_s, -1)
            po_u = self.conv_u(po_latent_u)
            po_s = self.conv_s(po_latent_s)
            L = torch.mean(po_d1)
            self.L_sum += L.item()
            self.L_count += 1
            L_use = L if self.L_count == 1 else self.L_avg
            pr_d1 = self.insnorm(pr_d1) * torch.abs(po_s) + L_use * po_u
            pr_d1 = self.pr_UpConv1(pr_d1)
            out = self.out_conv(pr_d1)
            return out, pr_u_dist, pr_s_dist, po_u_dist, po_s_dist, L

        else:
            pr_d3 = self.pr_UpConv3_1(pr_x3)
            pr_d2 = self.pr_Up2(pr_d3)
            height, width = pr_x2.size()[2:]
            pr_d2 = F.interpolate(pr_d2, size=(height, width), mode='bilinear', align_corners=False)
            pr_d2 = torch.cat((pr_x2, pr_d2), dim=1)
            pr_d2 = self.pr_UpConv2(pr_d2)
            pr_d2 = self.pr_UpConv2_1(pr_d2)
            pr_d1 = self.pr_Up1(pr_d2)
            height, width = pr_d1.size()[2:]
            pr_x1 = F.interpolate(pr_x1, size=(height, width), mode='bilinear', align_corners=False)
            pr_d1 = torch.cat((pr_x1, pr_d1), dim=1)
            pr_u_dist, pr_s_dist, _, _, _, _ = self.compute_z_pr(pr_d1)
            pr_latent_u = pr_u_dist.rsample()
            pr_latent_s = pr_s_dist.rsample()
            pr_latent_u = torch.unsqueeze(pr_latent_u, -1)
            pr_latent_u = torch.unsqueeze(pr_latent_u, -1)
            pr_latent_s = torch.unsqueeze(pr_latent_s, -1)
            pr_latent_s = torch.unsqueeze(pr_latent_s, -1)
            pr_u = self.conv_u(pr_latent_u)
            pr_s = self.conv_s(pr_latent_s)
            L_avg = self.get_L_avg()
            pr_d1 = self.insnorm(pr_d1) * torch.abs(pr_s) + L_avg * pr_u
            pr_d1 = self.pr_UpConv1(pr_d1)
            out = self.out_conv(pr_d1)

            return out


class PerceptionLoss(nn.Module):
    def __init__(self):
        super(PerceptionLoss, self).__init__()

        vgg = vgg19(pretrained=True)

        loss_network = nn.Sequential(*list(vgg.features)[:36]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()

    def forward(self, out_images, target_images):
        perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
        return perception_loss


class L1_Loss(nn.Module):
    def __init__(self):
        super(L1_Loss, self).__init__()

    def forward(self, output, target):
        loss = nn.L1Loss()(output, target)
        return loss


class L_Loss(nn.Module):
    def __init__(self):
        super(L_Loss, self).__init__()
    def forward(self, output, target):
        loss = nn.MSELoss()(output, target)
        return loss


class mynet(nn.Module):
    def __init__(self, opt):
        super(mynet, self).__init__()
        self.device = torch.device(opt.device)
        self.decoder = Decoder(device=self.device).to(self.device)
        self.criterion = nn.MSELoss().to(self.device)
        self.VGG19 = PerceptionLoss().to(self.device)
        self.L1_loss = L1_Loss().to(self.device)

    def forward(self, Input, label, training=True):
        self.Input = Input
        self.label = label

        if training:
            self.out, self.pr_u, self.pr_s, self.po_u, self.po_s, self.l = self.decoder.forward(self.Input, self.label)

    def sample(self, testing=False):
        if testing:
            self.out = self.decoder.forward(self.Input, self.label, training=False)
            return self.out

    def train_epoch(self, dataloader):
        self.decoder.reset_L()
        for Input, label in dataloader:
            self.forward(Input, label, training=True)
        L_avg = self.decoder.get_L_avg()
        return L_avg

    def kl_divergence(self, analytic=True):
        if analytic:
            kl_div_u = torch.mean(kl.kl_divergence(self.po_u, self.pr_u))
            kl_div_s = torch.mean(kl.kl_divergence(self.po_s, self.pr_s))
        return kl_div_u + kl_div_s

    def elbo(self, target, analytic_kl=True):
        self.kl_loss = self.kl_divergence(analytic=analytic_kl)
        self.reconstruction_loss = self.criterion(self.out, target)
        self.vgg19_loss = self.VGG19(self.out, target)
        self.l1_loss = self.L1_loss(self.out, target)
        return self.reconstruction_loss + self.vgg19_loss + self.kl_loss + self.l1_loss



