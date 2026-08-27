# full assembly of the sub-parts to form the complete net
import torch.nn.functional as F
import math
import random
import torch
import einops
import time
# from .block_utils import *
from timm.models.layers import trunc_normal_
from .swin_utils import *
from .swin_utils_3d import *

def positionalencoding1d(d_model, length, ratio):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1) * ratio
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe

class TVSRN3d_v1(nn.Module):
    def __init__(self):
        super(TVSRN3d_v1, self).__init__()

        ####################### Global config #######################
        img_size = opt.c_y
        mlp_ratio = opt.T3d_mlp
        resi_connection = opt.T_rc

        ######################## MAE Encoder ########################
        E_win = tuple(int(each) for each in opt.T3dE_w)
        E_depths = [opt.T3dE_d] * opt.T3dE_l
        E_num_heads = [opt.T3dE_n] * opt.T3dE_l

        E_num_in_ch = 1
        E_embed_dim = opt.T3dE_c
        E_out_c = 8

        self.Encoder = SwinTransformer3Dv3(in_chans=E_num_in_ch, 
                                           embed_dim=E_embed_dim,
                                           depths=E_depths, 
                                           num_heads=E_num_heads, 
                                           window_size=E_win,
                                           mlp_ratio=mlp_ratio,
                                           drop_path_rate=0)
        self.conv_after_E = nn.Conv3d(E_embed_dim, E_out_c, 3, 1, 1)

        # ######################## MAE MToken ########################
        self.c = E_out_c
        self.out_z = (opt.c_z - 1) * opt.ratio + 1

        self.positions_z = positionalencoding1d(self.c, self.out_z, 1).cuda()
        self.positions_z = self.positions_z.transpose(1,0).unsqueeze(2).unsqueeze(2).unsqueeze(0)

        # ######################## MAE Decoder ########################
        self.D_patch = opt.TD_p
        D_T_depths = [opt.TD_Td] * opt.TD_Tl
        D_T_num_heads = [opt.TD_n] * opt.TD_Tl

        D_I_depths = [opt.TD_Id] * opt.TD_Il
        D_T_num_heads = [opt.TD_n] * opt.TD_Il

        T_win = opt.TD_Tw
        I_win = opt.TD_Iw

        for i in range(1, opt.TD_s+1):
            T_embed = self.c * self.D_patch
            exec('''self.Decoder_T%s = Swin_backbone(img_size=(self.out_z, img_size), embed_dim=T_embed,
                                         depths=D_T_depths, num_heads=D_T_num_heads, window_size=T_win,
                                         mlp_ratio=mlp_ratio, resi_connection=resi_connection)''' % (i))
            exec('''del self.Decoder_T%s.conv_first''' % (i))
            exec('''del self.Decoder_T%s.conv_after_body''' % (i))

            I_embed = self.c * self.out_z
            exec('''self.Decoder_I%s = Swin_backbone(img_size=img_size, embed_dim=I_embed,
                                            depths=D_I_depths, num_heads=D_T_num_heads, window_size=I_win,
                                            mlp_ratio=mlp_ratio, resi_connection=resi_connection)''' % (i))
            exec('''del self.Decoder_I%s.conv_first''' % (i))
            exec('''del self.Decoder_I%s.conv_after_body''' % (i))

        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(I_embed, I_embed, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv_after_body = nn.Sequential(nn.Conv2d(I_embed, I_embed // 4, 3, 1, 1),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(I_embed // 4, I_embed // 4, 1, 1, 0),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(I_embed // 4, I_embed, 3, 1, 1))

        ######################## MAE Rec ########################
        self.conv_before_upsample = nn.Sequential(nn.Conv3d(self.c, 16, 1, 1, 0),
                                                  nn.LeakyReLU(inplace=True))
        self.conv_last = nn.Conv3d(16, 1, 1, 1, 0)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def cal_z(self, x, D):
        x_in = x.reshape(1, -1, opt.c_y, opt.c_x)
        x_out = D.forward_features(x_in)
        return x_out

    def cal_xy(self, x, D):
        x_in = x.reshape(-1, self.c, opt.c_y, opt.c_x)

        x_in_sag = einops.rearrange(x_in, 'dn c h (wn wp) -> wn (c wp) dn h', wp=self.D_patch)
        x_out_sag = D.forward_features(x_in_sag)
        x_out_sag = einops.rearrange(x_out_sag, 'wn (c wp) dn h -> dn c h (wn wp)', wp=self.D_patch)

        x_in_cor = einops.rearrange(x_in, 'dn c (hn hp) w -> hn (c hp) dn w', hp=self.D_patch)
        x_out_cor = D.forward_features(x_in_cor)
        x_out_cor = einops.rearrange(x_out_cor, 'hn (c hp) dn w -> dn c (hn hp) w', hp=self.D_patch)

        x_out = x_out_sag + x_out_cor
        return x_out

    def forward(self, x):
        x = x.squeeze().unsqueeze(0).unsqueeze(0)
        x = self.Encoder.conv_first(x)
        x_FE = self.Encoder.forward_features(x)

        x_Eout = self.Encoder.conv_after_body(x_FE) + x
        x_Eout = self.conv_after_E(x_Eout)

        B, C, D, H, W = x_Eout.shape
        x_up = F.upsample_nearest(x_Eout, size=(self.out_z, H, W))

        trans_input = x_up + self.positions_z
        trans_feature = trans_input.reshape(1, -1, opt.c_y, opt.c_x)

        for i in range(1, opt.TD_s + 1):
            trans_feature = eval('self.cal_xy(trans_feature, self.Decoder_T%s)' % i)
            trans_feature = eval('self.cal_z(trans_feature, self.Decoder_I%s)' % i)
        
        trans_output = self.conv_after_body(trans_feature) + trans_input.reshape(1, -1, opt.c_y, opt.c_x)
        trans_output = trans_output.reshape(1, self.c, -1, opt.c_y, opt.c_x)
        x_out = self.conv_last(self.conv_before_upsample(trans_output))

        return x_out[:, :, 3:-3]