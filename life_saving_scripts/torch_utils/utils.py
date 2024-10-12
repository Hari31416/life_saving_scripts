from life_saving_scripts.basic_utils import (
    create_simple_logger,
    BOLD,
    ITALIC,
    END,
    is_jupyter_notebook,
)

import torch
import torch.nn as nn
import torchvision
import numpy as np
from collections import OrderedDict
from typing import Dict, Union, List, Optional, Callable
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import display

logger = create_simple_logger(__name__)

A = np.ndarray
T = torch.Tensor
M = nn.Module


def get_model_tree(
    model: nn.Module,
) -> Dict[str, Union[nn.Module, Dict[str, nn.Module]]]:
    model_tree = OrderedDict()
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            model_tree[name] = get_model_tree(module)
        else:
            # if the module has no children, it is a leaf node, add it to the tree
            model_tree[name] = module

    return model_tree


def print_model_tree(
    model_tree: Union[Dict[str, Union[nn.Module, Dict[str, nn.Module]]], nn.Module],
    indent: int = 0,
    add_modules: bool = False,
):
    if isinstance(model_tree, nn.Module):
        model_tree = get_model_tree(model_tree)

    for name, module in model_tree.items():
        ended = False
        print(" " * indent + f"{BOLD}{name}{END}:", end="")

        if isinstance(module, dict):
            if not ended:
                print()
            print_model_tree(module, indent + 2, add_modules=add_modules)
        else:
            if add_modules:
                print(f"{' ' * (indent+2)}{ITALIC}{module}{END}", end="")
        if not ended:
            print()


def visualize_grid_images(
    images: Union[List[np.ndarray], List[torch.Tensor]],
    function_to_apply: Optional[Callable] = None,
    title_prefix: str = "",
) -> plt.Figure:
    if isinstance(images[0], torch.Tensor):
        images = [img.squeeze().cpu().detach().numpy() for img in images]
        # permute the channels to the last dimension
        images = [np.transpose(img, (1, 2, 0)) for img in images]
    num_images = len(images)
    columns = 5
    rows = int(np.ceil(num_images / columns))
    fig, axs = plt.subplots(rows, columns, figsize=(5 * columns, 5 * rows))
    for i, ax in enumerate(axs.flatten()):
        if i >= num_images:
            continue

        image = images[i]
        if function_to_apply:
            image = function_to_apply(image)
        image = np.clip(image, 0, 1)
        # plot the image
        ax.imshow(image)
        if title_prefix:
            ax.set_title(f"{title_prefix} {i}")
        ax.axis("off")
    plt.tight_layout()
    fig.show()
    return fig


def create_grid_image_from_tensor_images(
    images: List[torch.Tensor], nrow: int = 5
) -> torch.Tensor:
    grid_image = torchvision.utils.make_grid(images, nrow=nrow)
    return grid_image


class ImagePlotter:
    """A class to display images. It can be used to display images in a loop."""

    def __init__(
        self,
        cmap: str = "viridis",
        **kwargs: dict[str, any],
    ):
        """Initializes the ImagePlotter object.

        Parameters
        ----------
        title : str, optional
            The title of the figure. Default is "".
        cmap : str, optional
            The colormap to be used. Default is "viridis".
        kwargs
            Additional keyword arguments to be passed to the `plt.subplots` method.
        """

        self.fig, self.ax = plt.subplots(**kwargs)
        self.im = None
        self.cmap = cmap

    def update_image(
        self,
        image: Union[np.ndarray, Image.Image, T],
        title: str = "",
        path_to_save: Union[str, None] = None,
    ) -> None:
        # convert pil image to numpy array
        if isinstance(image, Image.Image):
            image = np.array(image)
        # convert tensor to numpy array
        if isinstance(image, torch.Tensor):
            image = image.squeeze().detach().cpu().numpy()
            # permute the channels to the last dimension
            image = np.transpose(image, (1, 2, 0))

        channels = image.shape[-1]
        if channels == 1 and self.cmap not in ["gray", "Greys"]:
            cmap = "gray"
        else:
            cmap = self.cmap

        if self.im is None:
            self.im = self.ax.imshow(image, cmap=cmap)
        else:
            self.im.set_data(image)
        self.ax.set_title(title)
        self.ax.title.set_fontsize(15)
        self.ax.axis("off")
        plt.draw()
        plt.pause(0.01)
        # display the figure if running in a Jupyter notebook
        if is_jupyter_notebook():
            display(self.fig, clear=True)
        if path_to_save is not None:
            self.fig.savefig(path_to_save)
