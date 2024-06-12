import math
import torch


def weights_init(net, init_type='xavier', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s type' % init_type)
    net.apply(init_func)


def get_xy(size):
    x0 = torch.ceil(torch.Tensor([size / 2])[0])
    y0 = torch.ceil(torch.Tensor([size / 2])[0])
    y, x = torch.meshgrid(
        [
            torch.linspace(-x0 + 1, x0 - 1, size),
            torch.linspace(-y0 + 1, y0 - 1, size),
        ], indexing='xy'
    )
    return x, y


def get_rox_xy(x, y, out_channl, thetas):
    fea_x, fea_y = x.repeat(out_channl, 1, 1), y.repeat(out_channl, 1, 1)
    rotx = fea_x * torch.cos(thetas.unsqueeze(1)) + fea_y * torch.sin(thetas.unsqueeze(1))
    roty = -fea_x * torch.sin(thetas.unsqueeze(1)) + fea_y * torch.cos(thetas.unsqueeze(1))
    return rotx, roty


def gs_weight(out_channl, sigmas, size=11):
    x, y = get_xy(size)
    fea_x, fea_y = x.repeat(out_channl, 1, 1), y.repeat(out_channl, 1, 1)
    g = torch.exp(
        -(fea_x ** 2 + fea_y ** 2) /
        (2 * (sigmas.unsqueeze(1)) ** 2)
    )
    g = g / g.sum(dim=(1, 2)).unsqueeze(1).unsqueeze(1)
    filters = g.unsqueeze(1)
    return filters


def gsx_weight(out_channl, sigmas, thetas, scales, size=11):
    fea_x, fea_y = get_xy(size)
    rotx, roty = get_rox_xy(fea_x, fea_y, out_channl, thetas)
    g_x = - (rotx / sigmas.unsqueeze(1)) * torch.exp(
        -(rotx ** 2 / (2 * sigmas.unsqueeze(1) ** 2) + roty ** 2 / (
                2 * (sigmas.unsqueeze(1) * scales.unsqueeze(1)) ** 2))
    )
    filters = g_x.unsqueeze(1)
    return filters


def gsx2_weight(out_channl, sigmas, thetas, scales, size=11):
    fea_x, fea_y = get_xy(size)
    rotx, roty = get_rox_xy(fea_x, fea_y, out_channl, thetas)
    g_x = - (1 / sigmas.unsqueeze(1) ** 2) * (1 - (rotx ** 2 / sigmas.unsqueeze(1) ** 2)) \
          * torch.exp(
        -(rotx ** 2 / (2 * sigmas.unsqueeze(1) ** 2) + roty ** 2 / (
                2 * (scales.unsqueeze(1) * sigmas.unsqueeze(1)) ** 2)))
    filters = g_x.unsqueeze(1)
    return filters


def LOG_weight(out_channl, sigmas, size=11):
    fea_x, fea_y = get_xy(size)
    fea_x, fea_y = fea_x.repeat(out_channl, 1, 1), fea_y.repeat(out_channl, 1, 1)
    g = (-1 / (math.pi * sigmas.unsqueeze(1) ** 4)) * \
        (1 - ((fea_x ** 2 + fea_y ** 2) / (2 * sigmas.unsqueeze(1) ** 2))) * \
        torch.exp(-(fea_x ** 2 + fea_y ** 2) / (2 * sigmas.unsqueeze(1) ** 2))
    filters = g.unsqueeze(1)
    return filters


def gab_weight(out_channl, sigmas, thetas, scales, freq, psi, size=11):
    fea_x, fea_y = get_xy(size)
    rotx, roty = get_rox_xy(fea_x, fea_y, out_channl, thetas)
    gb = torch.exp(
        -0.5 * (rotx ** 2 / sigmas.unsqueeze(1) ** 2 + roty ** 2 / (
                sigmas.unsqueeze(1) * scales.unsqueeze(1)) ** 2)
    ) * torch.cos(freq.unsqueeze(1) * rotx + psi.unsqueeze(1))
    filters = gb.unsqueeze(1)
    return filters


def Sechmid_weight(out_channl, sigmas, tao, size=11):
    fea_x, fea_y = get_xy(size)
    fea_x, fea_y = fea_x.repeat(out_channl, 1, 1), fea_y.repeat(out_channl, 1, 1)
    r = torch.sqrt(fea_x ** 2 + fea_y ** 2).cuda()
    g = torch.cos((r * math.pi * tao.unsqueeze(1)) / sigmas.unsqueeze(1)) * \
        torch.exp(-r ** 2 / (2 * sigmas.unsqueeze(1) ** 2))
    g = g - g.mean(dim=(1, 2)).unsqueeze(1).unsqueeze(1)
    g = g / torch.abs(g).sum(dim=(1, 2)).unsqueeze(1).unsqueeze(1)
    filters = g.unsqueeze(1)
    return filters
