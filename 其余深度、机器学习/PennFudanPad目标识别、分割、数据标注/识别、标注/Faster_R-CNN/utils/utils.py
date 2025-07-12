import numpy as np
from PIL import Image, ImageDraw




def relu(x):
    return max(x, 0)


def collate_fn(batch):
    return tuple(zip(*batch))
