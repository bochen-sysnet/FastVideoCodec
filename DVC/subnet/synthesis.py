#!/Library/Frameworks/Python.framework/Versions/3.5/bin/python3.5  
from .basics import *
import pickle
import os
import codecs
from .analysis import Analysis_net

class Synthesis_net(nn.Module):
    '''
    Decode residual
    '''
    def __init__(self, useAttn = False, channels=None, useRec=False, useDM=False):
        super(Synthesis_net, self).__init__()
        if channels is None:
            in_channels = out_channel_M
            conv_channels = out_channel_N
        else:
            in_channels = conv_channels = channels
        self.deconv1 = nn.ConvTranspose2d(in_channels,  conv_channels, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv1.weight.data, (math.sqrt(2 * 1 * (conv_channels +  in_channels) / (in_channels + in_channels))))
        torch.nn.init.constant_(self.deconv1.bias.data, 0.01)
        self.igdn1 = GDN( conv_channels, inverse=True)
        self.deconv2 = nn.ConvTranspose2d( conv_channels,  conv_channels, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv2.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv2.bias.data, 0.01)
        self.igdn2 = GDN( conv_channels, inverse=True)
        self.deconv3 = nn.ConvTranspose2d( conv_channels,  conv_channels, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv3.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv3.bias.data, 0.01)
        self.igdn3 = GDN( conv_channels, inverse=True)
        self.deconv4 = nn.ConvTranspose2d( conv_channels, 3, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv4.weight.data, (math.sqrt(2 * 1 * ( conv_channels + 3) / ( conv_channels +  conv_channels))))
        torch.nn.init.constant_(self.deconv4.bias.data, 0.01)
        if useAttn:
            self.layers = nn.ModuleList([])
            depth = 12
            for _ in range(depth):
                ff = FeedForward(in_channels)
                s_attn = Attention(in_channels, dim_head = 64, heads = 8)
                t_attn = Attention(in_channels, dim_head = 64, heads = 8)
                t_attn, s_attn, ff = map(lambda t: PreNorm(in_channels, t), (t_attn, s_attn, ff))
                self.layers.append(nn.ModuleList([t_attn, s_attn, ff]))
            self.frame_rot_emb = RotaryEmbedding(64)
            self.image_rot_emb = AxialRotaryEmbedding(64)
        self.useAttn = useAttn
        self.useRec = useRec
        if self.useRec:
            self.lstm = ConvLSTM(conv_channels)
        self.useDM = useDM
        if self.useDM:
            self.dm1 = DMBlock(in_channels)
            self.dm2 = DMBlock(conv_channels)
            self.dm3 = DMBlock(conv_channels)
        
    def forward(self, x):
        if self.useAttn:
            # B,C,H,W->1,BHW,C
            B,C,H,W = x.size()
            frame_pos_emb = self.frame_rot_emb(B,device=x.device)
            image_pos_emb = self.image_rot_emb(H,W,device=x.device)
            x = x.permute(0,2,3,1).reshape(1,-1,C).contiguous()
            for (t_attn, s_attn, ff) in self.layers:
                x = t_attn(x, 'b (f n) d', '(b n) f d', n = H*W, rot_emb = frame_pos_emb) + x
                x = s_attn(x, 'b (f n) d', '(b f) n d', f = B, rot_emb = image_pos_emb) + x
                x = ff(x) + x
            x = x.view(B,H,W,C).permute(0,3,1,2).contiguous()
        if self.useDM:
            x = self.dm1(x)
        x = self.igdn1(self.deconv1(x))
        if self.useDM:
            x = self.dm2(x)
        x = self.igdn2(self.deconv2(x))
        if self.useRec:
            x, self.hidden = self.lstm(x, self.hidden.to(x.device))
        if self.useDM:
            x = self.dm3(x)
        x = self.igdn3(self.deconv3(x))
        x = self.deconv4(x)
        return x

    def init_hidden(self, x):
        h,w = x.shape[:2]
        self.hidden = torch.zeros(1,out_channel_N*2,h//4,w//4)


class Synthesis_DM(nn.Module):
    def __init__(self):
        super(Synthesis_DM, self).__init__()
        in_channels = 96
        conv_channels = 64
        self.blocks = []
        self.blocks.append(DMBlock(in_channels))
        self.blocks.append(TransitionBlock(in_channels,  conv_channels, kernel_size=3, stride=2, padding=1, output_padding=1, deconv=True))
        self.blocks.append(DMBlock(conv_channels))
        self.blocks.append(TransitionBlock(conv_channels,  conv_channels, kernel_size=3, stride=2, padding=1, output_padding=1, deconv=True))
        self.blocks.append(DMBlock(conv_channels))
        self.blocks.append(TransitionBlock(conv_channels,  conv_channels, kernel_size=3, stride=1, padding=1, output_padding=1, deconv=True))
        self.blocks.append(TransitionBlock(conv_channels,  32, kernel_size=1, stride=1, padding=0, output_padding=0, deconv=True))
        self.blocks.append(TransitionBlock(32,  32, kernel_size=3, stride=2, padding=1, output_padding=1, deconv=True))
        self.blocks.append(TransitionBlock(32,  3, kernel_size=3, stride=2, padding=1, output_padding=1, deconv=True))
        self.blocks = nn.Sequential(self.blocks)

    def forward(self, x):
        return self.blocks(x)

def build_model():
    input_image = torch.zeros([7,3,256,256])
    analysis_net = Analysis_net()
    synthesis_net = Synthesis_net()
    feature = analysis_net(input_image)
    recon_image = synthesis_net(feature)

    print("input_image : ", input_image.size())
    print("feature : ", feature.size())
    print("recon_image : ", recon_image.size())



if __name__ == '__main__':
  build_model()
