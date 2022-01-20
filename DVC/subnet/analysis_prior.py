#!/Library/Frameworks/Python.framework/Versions/3.5/bin/python3.5  
from .basics import *
# import pickle
# import os
# import codecs
from .analysis import Analysis_net



class Analysis_prior_net(nn.Module):
    '''
    Compress residual prior
    '''
    def __init__(self, useAttn=False):
        super(Analysis_prior_net, self).__init__()
        self.conv1 = nn.Conv2d(out_channel_M, out_channel_N, 3, stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.conv1.weight.data, (math.sqrt(2 * (out_channel_M + out_channel_N) / (out_channel_M + out_channel_M))))
        torch.nn.init.constant_(self.conv1.bias.data, 0.01)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv2.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv2.bias.data, 0.01)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv3.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv3.bias.data, 0.01)
        if useAttn:
            self.layers = nn.ModuleList([])
            depth = 12
            for _ in range(depth):
                ff = FeedForward(out_channel_N)
                s_attn = Attention(out_channel_N, dim_head = 64, heads = 8)
                t_attn = Attention(out_channel_N, dim_head = 64, heads = 8)
                t_attn, s_attn, ff = map(lambda t: PreNorm(out_channel_N, t), (t_attn, s_attn, ff))
                self.layers.append(nn.ModuleList([t_attn, s_attn, ff]))
            self.frame_rot_emb = RotaryEmbedding(64)
            self.image_rot_emb = AxialRotaryEmbedding(64)
        self.useAttn = useAttn

    def forward(self, x):
        x = torch.abs(x)
        x = self.relu1(self.conv1(x))
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
        x = self.relu2(self.conv2(x))
        return self.conv3(x)


def build_model():
    input_image = torch.zeros([5, 3, 256, 256])
    analysis_net = Analysis_net()
    analysis_prior_net = Analysis_prior_net()

    feature = analysis_net(input_image)
    z = analysis_prior_net(feature)
    
    print(input_image.size())
    print(feature.size())
    print(z.size())



if __name__ == '__main__':
  build_model()
