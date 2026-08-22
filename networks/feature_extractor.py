import torch
import torch.nn as nn
import torchvision.models as models  # 导入 torchvision.models

class FeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super(FeatureExtractor, self).__init__()
        # 加载预训练的 SwinTransformer 模型
        self.model = models.swin_s(pretrained=pretrained)  # 假设 torchvision.models 有 swin_transformer_v2
        
        # 去掉模型的分类头，以便只保留特征提取部分
        self.features = nn.Sequential(*list(self.model.children())[:-1])
        
    def forward(self, x):
        # 提取特征
        x = self.features(x)
        # 将特征展平成一维向量
        x = torch.flatten(x, 1)
        return x

