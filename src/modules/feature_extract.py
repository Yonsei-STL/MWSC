import torch
import torch.nn as nn
import numpy as np
from gl_clip import clip

class Feature_Extractor(nn.Module):
    def __init__(self, device, model_name):
        super().__init__()
        self.model_name = model_name
        self.model, self.preprocess = clip.load(self.model_name, device)
        
        if self.model_name == 'ViT-B/16' or self.model_name == 'ViT-L/14':
            self.pool2d = torch.nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.pool2d = None
        
        for param in self.model.parameters():
            param.requires_grad = False

    def downsample_feature(self, x):
        B = x.size(0)
        l_d = x.size(2)
        s = int(np.sqrt(x.size(1)))
        assert x.size(1) == (s * s)
        x = x.permute(0, 2, 1)
        x = x.view(B, l_d, s, s).contiguous()
        x = self.pool2d(x)
        x = x.reshape(B, l_d, -1)
        x = x.permute(0, 2, 1)
        return x
        
    def forward(self, images, prompts):
        with torch.no_grad():
            text_features = self.model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            local_feat, global_feat = self.model.encode_image(images)

            global_feat = global_feat / global_feat.norm(dim=-1, keepdim=True)
            local_feat = local_feat / local_feat.norm(dim=-1, keepdim=True)
 
            if self.pool2d:
                local_feat = self.downsample_feature(local_feat)

        return global_feat, local_feat, text_features