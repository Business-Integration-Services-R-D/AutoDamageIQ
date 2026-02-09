import torch
import matplotlib.pyplot as plt


def show_image_and_mask(image: torch.Tensor, mask: torch.Tensor, title: str = "Damage Overlay"):
    """
    image: [3, H, W], 0-1 arası
    mask:  [1, H, W], 0-1 arası
    """
    img = image.detach().cpu().numpy()
    msk = mask.detach().cpu().numpy()

    img = img.transpose(1, 2, 0)  # CHW -> HWC
    msk = msk[0]                  # [H, W]

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(img)
    plt.imshow(msk, alpha=0.5, cmap="Reds")
    plt.title(title)
    plt.axis("off")

    plt.tight_layout()
    plt.show()
