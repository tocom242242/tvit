from torch import nn, Tensor
from einops.layers.torch import Rearrange, Reduce

# class ClassificationHead(nn.Module):
#     def __init__(self, emb_size:int = 768, n_classes : int = 3):
#         super().__init__()

#     def forward(self, x:Tensor):
#         return x


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size: int = 768, n_classes: int = 1000):
        super().__init__(
            Reduce("b n e -> b e", reduction="mean"),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes),
        )
