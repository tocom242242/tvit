import torch
from torch import nn
from vit.head import ClassificationHead
from vit.encoder import TransformerEncoder
from vit.patch_embedding import PatchEmbedding


class ViT(nn.Sequential):
    def __init__(
        self,
        in_channels: int = 3,
        patch_size: int = 16,
        emb_size: int = 768,
        img_size: int = 224,
        depth: int = 1,
        n_classes: int = 10,
        **kwargs
    ):
        super().__init__(
            PatchEmbedding(in_channels, patch_size, emb_size, img_size),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size, n_classes),
        )


def main():
    print("start")
    _vit = ViT()
    x = torch.rand(1, 3, 224, 224)
    import IPython

    IPython.embed(header="")
    print("end")


if __name__ == "__main__":
    print("start")
    main()
    print("end")
