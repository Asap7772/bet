import torchvision
import torch.nn as nn
import utils
from einops import rearrange


class resnet18(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        freeze_pretrained: bool = True,
        output_dim: int = 512,  # fixed for resnet18; included for consistency with config
        reproject_dim: int = 519,
    ):
        super().__init__()
        resnet = torchvision.models.resnet18(pretrained=pretrained)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.flatten = nn.Flatten()
        self.pretrained = pretrained
        self.freeze_pretrained = pretrained and freeze_pretrained
        if self.freeze_pretrained:
            utils.freeze_module(self.resnet)
        self.normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.linear_proj = nn.Linear(reproject_dim, output_dim)

    def forward(self, x):
        # if NTCHW, flatten to NCHW first
        is_seq = x.dim() == 5
        if is_seq:
            n = x.shape[0]
            t = x.shape[1]
            x = rearrange(x, "n t c h w -> (n t) c h w")
        x = self.normalize(x)
        out = self.resnet(x)
        out = self.flatten(out)
        if is_seq:
            out = rearrange(out, "(n t) e -> n t e", n=n, t=t)
        return out

class resnet34(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        freeze_pretrained: bool = True,
        output_dim: int = 512,  # fixed for resnet18; included for consistency with config
        reproject_dim: int = 519,
    ):
        super().__init__()
        resnet = torchvision.models.resnet34(pretrained=pretrained)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.flatten = nn.Flatten()
        self.pretrained = pretrained
        self.freeze_pretrained = pretrained and freeze_pretrained
        if self.freeze_pretrained:
            utils.freeze_module(self.resnet)
        self.normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.linear_proj = nn.Linear(reproject_dim, output_dim)

    def forward(self, x):
        # if NTCHW, flatten to NCHW first
        is_seq = x.dim() == 5
        if is_seq:
            n = x.shape[0]
            t = x.shape[1]
            x = rearrange(x, "n t c h w -> (n t) c h w")
        x = self.normalize(x)
        out = self.resnet(x)
        out = self.flatten(out)
        if is_seq:
            out = rearrange(out, "(n t) e -> n t e", n=n, t=t)
        return out