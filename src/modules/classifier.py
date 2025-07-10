import torch.nn as nn
import torch
from collections import OrderedDict

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class MWSC(nn.Module):
    def __init__(self, model_name, num_weather_types, num_severity_levels, ablation_mode):
        super(MWSC, self).__init__()
        self.ablation_mode = ablation_mode
        self.num_weather_types = num_weather_types
        self.num_severity_levels = num_severity_levels
        
        if model_name == 'ViT-L/14':
            gi_t_dim = 768
            li_dim = 1024
        else:
            gi_t_dim = 512
            li_dim = 768    
            
        if ablation_mode == 1:
            self.gllattn = CrossAttention(gi_t_dim, li_dim)
            self.t2i_attn = CrossAttention(gi_t_dim, gi_t_dim)
            self.i2t_attn = CrossAttention(gi_t_dim, gi_t_dim)
            self.feedforward = nn.Sequential(
                nn.Linear(gi_t_dim, 2048),
                nn.ReLU(),
                nn.Dropout(0.0),
                nn.Linear(2048, gi_t_dim)
            )
            self.weather_mlp = nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(gi_t_dim * num_weather_types, gi_t_dim * 4)),
                ("gelu", QuickGELU()),
                ("c_proj", nn.Linear(gi_t_dim * 4, num_weather_types))
            ]))
            self.severity_mlp = nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(gi_t_dim * num_severity_levels, gi_t_dim * 4)),
                ("gelu", QuickGELU()),
                ("c_proj", nn.Linear(gi_t_dim * 4, num_severity_levels))
            ]))
            
        elif ablation_mode == 2:
            self.weather_mlp = nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(num_weather_types, gi_t_dim * 4)),
                ("gelu", QuickGELU()),
                ("c_proj", nn.Linear(gi_t_dim * 4, num_weather_types))
            ]))
            self.severity_mlp = nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(num_severity_levels, gi_t_dim * 4)),
                ("gelu", QuickGELU()),
                ("c_proj", nn.Linear(gi_t_dim * 4, num_severity_levels))
            ]))
            
        elif ablation_mode == 3:
            self.gllattn = CrossAttention(gi_t_dim, li_dim)
            self.feedforward = nn.Sequential(
                nn.Linear(gi_t_dim, 2048),
                nn.ReLU(),
                nn.Dropout(0.0),
                nn.Linear(2048, gi_t_dim)
            )
            self.weather_mlp = nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(num_weather_types, gi_t_dim * 4)),
                ("gelu", QuickGELU()),
                ("c_proj", nn.Linear(gi_t_dim * 4, num_weather_types))
            ]))
            self.severity_mlp = nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(num_severity_levels, gi_t_dim * 4)),
                ("gelu", QuickGELU()),
                ("c_proj", nn.Linear(gi_t_dim * 4, num_severity_levels))
            ]))
            
        elif ablation_mode == 4:
            self.t2i_attn = CrossAttention(gi_t_dim, gi_t_dim)
            self.i2t_attn = CrossAttention(gi_t_dim, gi_t_dim)
            self.feedforward = nn.Sequential(
                nn.Linear(gi_t_dim, 2048),
                nn.ReLU(),
                nn.Dropout(0.0),
                nn.Linear(2048, gi_t_dim)
            )
            self.weather_mlp = nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(gi_t_dim * num_weather_types, gi_t_dim * 4)),
                ("gelu", QuickGELU()),
                ("c_proj", nn.Linear(gi_t_dim * 4, num_weather_types))
            ]))
            self.severity_mlp = nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(gi_t_dim * num_severity_levels, gi_t_dim * 4)),
                ("gelu", QuickGELU()),
                ("c_proj", nn.Linear(gi_t_dim * 4, num_severity_levels))
            ]))
         
    def forward(self, global_feat, local_feat, text_features):
        if self.ablation_mode == 1:
            global_feat = global_feat.unsqueeze(1)
            gll = self.gllattn(global_feat.float(), local_feat.float(), local_feat.float())
            gll = gll + global_feat.float()
            gll = gll / gll.norm(dim=-1, keepdim=True)
            ff_out = self.feedforward(gll)
            gll = gll + ff_out
            gll = gll / gll.norm(dim=-1, keepdim=True)


            text_features = text_features.unsqueeze(0)
            t2i = self.t2i_attn(text_features.float(), gll, gll)
            t2i = t2i + text_features.float()
            t2i = t2i / t2i.norm(dim=-1, keepdim=True)
            ff_out = self.feedforward(t2i)
            t2i = t2i + ff_out
            t2i = t2i / t2i.norm(dim=-1, keepdim=True)
            
            t2i_weather_attn = t2i[:, :self.num_weather_types, :]
            t2i_weather_attn = t2i_weather_attn.permute(1, 2, 0)
            t2i_weather_attn = t2i_weather_attn.view(-1, t2i_weather_attn.size(-1)).contiguous()
            t2i_weather_attn = t2i_weather_attn.transpose(-2, -1)
            t2i_severity_attn = t2i[:, self.num_weather_types:, :]
            t2i_severity_attn = t2i_severity_attn.permute(1, 2, 0)
            t2i_severity_attn = t2i_severity_attn.view(-1, t2i_severity_attn.size(-1)).contiguous()
            t2i_severity_attn = t2i_severity_attn.transpose(-2, -1)
            
            
            text_features = text_features.transpose(0, 1)
            gll = gll.transpose(0, 1)
            i2t = self.i2t_attn(gll, text_features.float(), text_features.float())
            i2t = i2t + gll
            i2t = i2t / i2t.norm(dim=-1, keepdim=True)
            ff_out = self.feedforward(i2t)
            i2t = i2t + ff_out
            i2t = i2t / i2t.norm(dim=-1, keepdim=True)
            
            i2t_weather_attn = i2t[:self.num_weather_types, :, :]
            i2t_weather_attn = i2t_weather_attn.permute(0, 2, 1) 
            i2t_weather_attn = i2t_weather_attn.contiguous().view(-1, i2t_weather_attn.size(-1))
            i2t_weather_attn = i2t_weather_attn.transpose(-2, -1)
            i2t_severity_attn = i2t[self.num_weather_types:, :, :]
            i2t_severity_attn = i2t_severity_attn.permute(0, 2, 1)
            i2t_severity_attn = i2t_severity_attn.contiguous().view(-1, i2t_severity_attn.size(-1))
            i2t_severity_attn = i2t_severity_attn.transpose(-2, -1)


            weather_attn = (t2i_weather_attn + i2t_weather_attn)/2
            severity_attn = (t2i_severity_attn + i2t_severity_attn)/2
            weather_out = self.weather_mlp(weather_attn)
            severity_out = self.severity_mlp(severity_attn)
            return weather_out, severity_out
        
        
        elif self.ablation_mode == 2:
            sim = (100 * global_feat.float() @ text_features.T.float())
            weather_sim = sim[:, :self.num_weather_types]
            severity_sim = sim[:, self.num_weather_types:]
            
            
            weather_out = self.weather_mlp(weather_sim)
            severity_out = self.severity_mlp(severity_sim)
            return weather_out, severity_out
        
        
        elif self.ablation_mode == 3:
            global_feat = global_feat.unsqueeze(1)
            gll = self.gllattn(global_feat.float(), local_feat.float(), local_feat.float())
            gll = gll + global_feat.float()
            gll = gll / gll.norm(dim=-1, keepdim=True)
            ff_out = self.feedforward(gll)
            gll = gll + ff_out
            gll = gll / gll.norm(dim=-1, keepdim=True)
            gll = gll.squeeze(1)


            sim = (100 * gll @ text_features.T.float())
            weather_sim = sim[:, :self.num_weather_types]
            severity_sim = sim[:, self.num_weather_types:]
            
            weather_out = self.weather_mlp(weather_sim)
            severity_out = self.severity_mlp(severity_sim)
            return weather_out, severity_out
        
        
        elif self.ablation_mode == 4:
            global_feat = global_feat.unsqueeze(1)
            text_features = text_features.unsqueeze(0)
            t2i = self.t2i_attn(text_features.float(), global_feat.float(), global_feat.float())
            t2i = t2i + text_features.float()
            t2i = t2i / t2i.norm(dim=-1, keepdim=True)
            ff_out = self.feedforward(t2i)
            t2i = t2i + ff_out
            t2i = t2i / t2i.norm(dim=-1, keepdim=True)

            t2i_weather_attn = t2i[:, :self.num_weather_types, :]
            t2i_weather_attn = t2i_weather_attn.permute(1, 2, 0)
            t2i_weather_attn = t2i_weather_attn.view(-1, t2i_weather_attn.size(-1)).contiguous()
            t2i_weather_attn = t2i_weather_attn.transpose(-2, -1)
            t2i_severity_attn = t2i[:, self.num_weather_types:, :]
            t2i_severity_attn = t2i_severity_attn.permute(1, 2, 0)
            t2i_severity_attn = t2i_severity_attn.view(-1, t2i_severity_attn.size(-1)).contiguous()
            t2i_severity_attn = t2i_severity_attn.transpose(-2, -1)


            text_features = text_features.transpose(0, 1)
            global_feat = global_feat.transpose(0, 1)
            i2t = self.i2t_attn(global_feat.float(), text_features.float(), text_features.float())
            i2t = i2t + global_feat.float()
            i2t = i2t / i2t.norm(dim=-1, keepdim=True)
            ff_out = self.feedforward(i2t)
            i2t = i2t + ff_out
            i2t = i2t / i2t.norm(dim=-1, keepdim=True)

            i2t_weather_attn = i2t[:self.num_weather_types, :, :]
            i2t_weather_attn = i2t_weather_attn.permute(0, 2, 1)
            i2t_weather_attn = i2t_weather_attn.contiguous().view(-1, i2t_weather_attn.size(-1))
            i2t_weather_attn = i2t_weather_attn.transpose(-2, -1)
            i2t_severity_attn = i2t[self.num_weather_types:, :, :]
            i2t_severity_attn = i2t_severity_attn.permute(0, 2, 1)
            i2t_severity_attn = i2t_severity_attn.contiguous().view(-1, i2t_severity_attn.size(-1))
            i2t_severity_attn = i2t_severity_attn.transpose(-2, -1)


            weather_attn = t2i_weather_attn + i2t_weather_attn
            severity_attn = t2i_severity_attn + i2t_severity_attn
            weather_out = self.weather_mlp(weather_attn)
            severity_out = self.severity_mlp(severity_attn)
            return weather_out, severity_out
        
        
class CrossAttention(nn.Module):
    def __init__(self, q_dim=512, kv_dim=768, qkv_bias=False):
        super().__init__()
        self.scale = kv_dim ** -0.5

        self.q = nn.Linear(q_dim, q_dim, bias=qkv_bias)
        self.k = nn.Linear(kv_dim, q_dim, bias=qkv_bias)
        self.v = nn.Linear(kv_dim, q_dim, bias=qkv_bias)
        self.proj = nn.Linear(q_dim, q_dim)
        self._reset_parameters()
        
        self.attn_weights = None
        
    def _reset_parameters(self):
        torch.manual_seed(0)
        nn.init.xavier_uniform_(self.q.weight)
        nn.init.xavier_uniform_(self.k.weight)
        nn.init.xavier_uniform_(self.v.weight)
        nn.init.xavier_uniform_(self.proj.weight)
        if self.k.bias is not None:
            nn.init.xavier_normal_(self.k.bias)
        if self.v.bias is not None:
            nn.init.xavier_normal_(self.v.bias)
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0.)
            
    def forward(self, x_q, x_k, x_v):
        q = self.q(x_q)
        k = self.k(x_k)
        v = self.v(x_v)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        self.attn_weights = attn.detach()

        x = (attn @ v)
        x = self.proj(x)

        return x