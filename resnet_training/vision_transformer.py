import timm
vit_small = timm.create_model(
    "deit_small_patch16_224",
    pretrained=True,
    num_classes=10
)
