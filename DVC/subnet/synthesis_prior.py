#!/Library/Frameworks/Python.framework/Versions/3.5/bin/python3.5  
from .basics import *
import pickle
import os
import codecs
from .analysis import Analysis_net
from .analysis_prior import Analysis_prior_net
from .synthesis import Synthesis_net


class Synthesis_prior_net(nn.Module):
    '''
    Decode residual prior
    '''
    def __init__(self, useAttn = False, channels=None, useRec=False, useDM=False):
        super(Synthesis_prior_net, self).__init__()
        if channels is None:
            conv_channels = out_channel_N
            out_channels = out_channel_M
        else:
            conv_channels = out_channels = channels
        self.deconv1 = nn.ConvTranspose2d(conv_channels, conv_channels, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv1.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv1.bias.data, 0.01)
        self.relu1 = nn.ReLU()
        self.deconv2 = nn.ConvTranspose2d(conv_channels, conv_channels, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv2.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv2.bias.data, 0.01)
        self.relu2 = nn.ReLU()
        self.deconv3 = nn.ConvTranspose2d(conv_channels, out_channels, 3, stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.deconv3.weight.data, (math.sqrt(2 * 1 * (conv_channels + out_channels) / (conv_channels + conv_channels))))
        torch.nn.init.constant_(self.deconv3.bias.data, 0.01)
        if useAttn:
            self.layers = nn.ModuleList([])
            depth = 12
            for _ in range(depth):
                ff = FeedForward(conv_channels)
                s_attn = Attention(conv_channels, dim_head = 64, heads = 8)
                t_attn = Attention(conv_channels, dim_head = 64, heads = 8)
                t_attn, s_attn, ff = map(lambda t: PreNorm(conv_channels, t), (t_attn, s_attn, ff))
                self.layers.append(nn.ModuleList([t_attn, s_attn, ff]))
            self.frame_rot_emb = RotaryEmbedding(64)
            self.image_rot_emb = AxialRotaryEmbedding(64)
        self.useAttn = useAttn
        self.useRec = useRec
        if self.useRec:
            self.lstm = ConvLSTM(conv_channels)
        self.useDM = useDM
        if self.useDM:
            self.dm = DMBlock(conv_channels)


    def forward(self, x):
        x = self.relu1(self.deconv1(x))
        if self.useRec:
            x, self.hidden = self.lstm(x, self.hidden.to(x.device))
        if self.useDM:
            x = self.dm(x)
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
        x = self.relu2(self.deconv2(x))
        return torch.exp(self.deconv3(x))

    def init_hidden(self, x):
        h,w = x.shape[:2]
        self.hidden = torch.zeros(1,out_channel_N*2,h//32,w//32)


def build_model():
      
    input_image = torch.zeros([7,3,256,256])
    analysis_net = Analysis_net()
    analysis_prior_net = Analysis_prior_net()
    synthesis_net = Synthesis_net()
    synthesis_prior_net = Synthesis_prior_net()

    feature = analysis_net(input_image)
    z = analysis_prior_net(feature)

    compressed_z = torch.round(z)

    recon_sigma = synthesis_prior_net(compressed_z)


    compressed_feature_renorm = feature / recon_sigma
    compressed_feature_renorm = torch.round(compressed_feature_renorm)
    compressed_feature_denorm = compressed_feature_renorm * recon_sigma

    recon_image = synthesis_net(compressed_feature_denorm)

    print("input_image : ", input_image.size())
    print("feature : ", feature.size())
    print("z : ", z.size())
    print("recon_sigma : ", recon_sigma.size())
    print("recon_image : ", recon_image.size())


if __name__ == '__main__':
    build_model()
