"""CTransPath factory (Wang et al. / Path2Space).

Prefer the bundled SwinTransformer (matches Zenodo ctranspath.pth key layout).
Fall back to timm only if needed (newer timm requires ConvStem **kwargs + BHWC).
"""

from __future__ import annotations

from itertools import repeat
import collections.abc

from torch import nn


def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable) and not isinstance(x, (str, bytes)):
        return x
    return tuple(repeat(x, 2))


class ConvStem(nn.Module):
    """
    Conv patch embed used by CTransPath.

    - layout='bnc': BCHW -> BNC (original TransPath / local SwinTransformer)
    - layout='bhwc': BCHW -> BHWC (timm >=0.9 Swin NHWC path)
    """

    def __init__(
        self,
        img_size=224,
        patch_size=4,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
        layout: str = "bnc",
        **kwargs,  # absorb timm extras e.g. output_fmt, strict_img_size
    ):
        super().__init__()
        assert patch_size == 4
        assert embed_dim % 8 == 0
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        # timm may pass output_fmt='NHWC'
        output_fmt = kwargs.get("output_fmt")
        if output_fmt is not None and str(output_fmt).upper() in {"NHWC", "BHWC"}:
            layout = "bhwc"
        self.layout = layout

        stem = []
        input_dim, output_dim = 3, embed_dim // 8
        for _ in range(2):
            stem.append(
                nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1, bias=False)
            )
            stem.append(nn.BatchNorm2d(output_dim))
            stem.append(nn.ReLU(inplace=True))
            input_dim = output_dim
            output_dim *= 2
        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], (
            f"Input image size ({H}*{W}) doesn't match model "
            f"({self.img_size[0]}*{self.img_size[1]})."
        )
        x = self.proj(x)
        if self.layout == "bhwc":
            x = x.permute(0, 2, 3, 1)  # BCHW -> BHWC
            x = self.norm(x)
            return x
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


def build_ctranspath_local(num_classes: int = 0) -> nn.Module:
    """Build CTransPath with bundled Swin (matches Zenodo ctranspath.pth)."""
    from .swin_transformer import SwinTransformer

    model = SwinTransformer(
        img_size=224,
        patch_size=4,
        in_chans=3,
        num_classes=num_classes if num_classes > 0 else 0,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        ape=False,
        patch_norm=True,
        embed_layer=ConvStem,
        use_checkpoint=False,
    )
    if num_classes == 0:
        model.head = nn.Identity()
    return model


def build_ctranspath_with_timm(num_classes: int = 0) -> nn.Module:
    """
    Prefer local architecture (weight-compatible).
    Fall back to timm Swin-Tiny + ConvStem (BHWC) if local import fails.
    """
    try:
        model = build_ctranspath_local(num_classes=num_classes)
        print("Built CTransPath with local SwinTransformer + ConvStem")
        return model
    except Exception as e:
        print(f"Local CTransPath build failed ({e}); trying timm fallback")

    try:
        import timm
    except ImportError as err:
        raise ImportError("timm is required for CTransPath fallback. pip install timm") from err

    def _convstem_timm(*args, **kwargs):
        kwargs.setdefault("layout", "bhwc")
        return ConvStem(*args, **kwargs)

    model = timm.create_model(
        "swin_tiny_patch4_window7_224",
        embed_layer=_convstem_timm,
        pretrained=False,
        num_classes=num_classes if num_classes > 0 else 0,
    )
    if num_classes == 0:
        model.head = nn.Identity()
    print("Built CTransPath with timm + ConvStem (BHWC)")
    return model
