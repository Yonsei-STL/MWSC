import torch.nn as nn
import timm
    
class Finetune_model(nn.Module):
    def __init__(self, model_name, num_weather_types, num_severity_levels, pretrain):
        super(Finetune_model, self).__init__()
        if pretrain == True:
            self.net = timm.create_model(model_name, pretrained=True)
        else:
            self.net = timm.create_model(model_name, pretrained=False)

        list1 = ['vit_base_patch32_224', 'vit_base_patch32_clip_224', 'beit_base_patch16_224', 'deit3_base_patch16_224', 'vitamin_base_224']
        list2 = ['resnet50']
        list3 = ['poolformerv2_m48', 'swinv2_cr_base_224', 'convnextv2_base', 'vgg16', 'mambaout_base']
        list4 = ['vit_base_patch14_dinov2', 'vit_base_patch16_siglip_224']
        list5 = ['crossvit_18_240']
        list6 = ['caformer_b36', 'convformer_b36']
        
        if model_name in list1:
            out_features = self.net.head.out_features
        elif model_name in list2:
            out_features = self.net.fc.out_features
        elif model_name in list3:
            out_features = self.net.head.fc.out_features
        elif model_name in list4:
            out_features = self.net.num_features
        elif model_name in list5:
            out_features = self.net.head[0].out_features
        elif model_name in list6:
            out_features = self.net.head.fc.fc2.out_features
        else:
            print("Mismatched models.")
            
        self.weather_fc = nn.Linear(out_features, num_weather_types)
        self.severity_fc = nn.Linear(out_features, num_severity_levels)
        
    def forward(self, x):   
        x = self.net(x)
        weather_out = self.weather_fc(x)
        severity_out = self.severity_fc(x)
        return weather_out, severity_out
