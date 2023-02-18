#!/Library/Frameworks/Python.framework/Versions/3.5/bin/python3.5
from .basics import *
import pickle
import os
import codecs
from .analysis_mv import Analysis_mv_net
# gdn = tf.contrib.layers.gdn

class Synthesis_mv_net(nn.Module):
    '''
    Compress motion
    '''
    def __init__(self, useAttn=False, channels=None, useRec=False, useMod=False):
        super(Synthesis_mv_net, self).__init__()
        if channels is None:
            in_channels = conv_channels = out_channel_mv
        else:
            conv_channels = out_channel_mv
            in_channels = channels
        self.deconv1 = nn.ConvTranspose2d(in_channels,  conv_channels, 3, stride=2, padding=1, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv1.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv1.bias.data, 0.01)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1)
        self.deconv2 = nn.Conv2d( conv_channels,  conv_channels, 3, stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.deconv2.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv2.bias.data, 0.01)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1)
        self.deconv3 = nn.ConvTranspose2d( conv_channels,  conv_channels, 3, stride=2, padding=1, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv3.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv3.bias.data, 0.01)
        self.relu3 = nn.LeakyReLU(negative_slope=0.1)
        self.deconv4 = nn.Conv2d( conv_channels,  conv_channels, 3, stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.deconv4.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv4.bias.data, 0.01)
        self.relu4 = nn.LeakyReLU(negative_slope=0.1)
        self.deconv5 = nn.ConvTranspose2d( conv_channels,  conv_channels, 3, stride=2, padding=1, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv5.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv5.bias.data, 0.01)
        self.relu5 = nn.LeakyReLU(negative_slope=0.1)
        self.deconv6 = nn.Conv2d( conv_channels,  conv_channels, 3, stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.deconv6.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv6.bias.data, 0.01)
        self.relu6 = nn.LeakyReLU(negative_slope=0.1)
        self.deconv7 = nn.ConvTranspose2d( conv_channels,  conv_channels, 3, stride=2, padding=1, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv7.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv7.bias.data, 0.01)
        self.relu7 = nn.LeakyReLU(negative_slope=0.1)
        self.deconv8 = nn.Conv2d( conv_channels, 2, 3, stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.deconv8.weight.data, (math.sqrt(2 * 1 * ( conv_channels + 2) / ( conv_channels +  conv_channels))))
        torch.nn.init.constant_(self.deconv8.bias.data, 0.01)
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
        self.useMod = useMod
        if useMod:
            self.mod = Modulate()
        
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
        x = self.relu1(self.deconv1(x))
        if self.useMod: x = self.mod(x,level)
        x = self.relu2(self.deconv2(x))
        if self.useMod: x = self.mod(x,level)
        x = self.relu3(self.deconv3(x))
        if self.useMod: x = self.mod(x,level)
        x = self.relu4(self.deconv4(x))
        if self.useMod: x = self.mod(x,level)
        if self.useRec:
            x, self.hidden = self.lstm(x, self.hidden.to(x.device))
        x = self.relu5(self.deconv5(x))
        if self.useMod: x = self.mod(x,level)
        x = self.relu6(self.deconv6(x))
        if self.useMod: x = self.mod(x,level)
        x = self.relu7(self.deconv7(x))
        if self.useMod: x = self.mod(x,level)
        x = self.deconv8(x)
        if self.useMod: x = self.mod(x,level)
        return x

    def init_hidden(self, x):
        h,w = x.shape[:2]
        self.hidden = torch.zeros(1,out_channel_mv*2,h//4,w//4)


def build_model():
    input_image = torch.zeros([4, 2, 256, 256])
    analysis_mv_net = Analysis_mv_net()
    synthesis_mv_net = Synthesis_mv_net()
    feature = analysis_mv_net(input_image)
    recon_image = synthesis_mv_net(feature)
    print(input_image.size())
    print(feature.size())
    print(recon_image.size())
    



if __name__ == '__main__':
    build_model()
