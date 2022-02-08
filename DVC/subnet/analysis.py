#!/Library/Frameworks/Python.framework/Versions/3.5/bin/python3.5
from .basics import *
import pickle
import os
import codecs

# gdn = tf.contrib.layers.gdn


class Analysis_net(nn.Module):
    '''
    Compress residual
    '''
    def __init__(self, useAttn=False, channels=None):
        super(Analysis_net, self).__init__()
        if channels is None:
            conv_channels = out_channel_N
            out_channels = out_channel_M
        else:
            conv_channels = out_channels = channels
        self.conv1 = nn.Conv2d(3,  conv_channels, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv1.weight.data, (math.sqrt(2 * (3 +  conv_channels) / (6))))
        torch.nn.init.constant_(self.conv1.bias.data, 0.01)
        self.gdn1 = GDN( conv_channels)
        self.conv2 = nn.Conv2d( conv_channels,  conv_channels, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv2.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv2.bias.data, 0.01)
        self.gdn2 = GDN( conv_channels)
        self.conv3 = nn.Conv2d( conv_channels,  conv_channels, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv3.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv3.bias.data, 0.01)
        self.gdn3 = GDN( conv_channels)
        self.conv4 = nn.Conv2d( conv_channels, out_channels, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv4.weight.data, (math.sqrt(2 * (out_channels +  conv_channels) / ( conv_channels +  conv_channels))))
        torch.nn.init.constant_(self.conv4.bias.data, 0.01)
        if useAttn:
            self.layers = nn.ModuleList([])
            depth = 12
            for _ in range(depth):
                ff = FeedForward(channels)
                s_attn = Attention(channels, dim_head = 64, heads = 8)
                t_attn = Attention(channels, dim_head = 64, heads = 8)
                t_attn, s_attn, ff = map(lambda t: PreNorm(channels, t), (t_attn, s_attn, ff))
                self.layers.append(nn.ModuleList([t_attn, s_attn, ff]))
            self.frame_rot_emb = RotaryEmbedding(64)
            self.image_rot_emb = AxialRotaryEmbedding(64)
        self.useAttn = useAttn

    def forward(self, x):
        x = self.gdn1(self.conv1(x))
        x = self.gdn2(self.conv2(x)) 
        x = self.gdn3(self.conv3(x))
        x = self.conv4(x)
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
        return x


def build_model():
        input_image = Variable(torch.zeros([4, 3, 256, 256]))

        analysis_net = Analysis_net()
        feature = analysis_net(input_image)

        print(feature.size())
        # feature = sess.run(weights)

        # print(weights_val)

        # gamma_val = sess.run(gamma)

        # print(gamma_val)


if __name__ == '__main__':
    build_model()
