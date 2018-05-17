import torch.utils.data as data
from PIL import Image
import numpy as np
import h5py

class Datafeed(data.Dataset):

    def __init__(self, x_train, y_train, transform=None):
        self.x_train = x_train
        self.y_train = y_train
        self.transform = transform

    def __getitem__(self, index):
        img = self.x_train[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, self.y_train[index]

    def __len__(self):
        return len(self.x_train)

class DatafeedImage(data.Dataset):
    def __init__(self, x_train, y_train, transform=None):
        self.x_train = x_train
        self.y_train = y_train
        self.transform = transform

    def __getitem__(self, index):
        img = self.x_train[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, self.y_train[index]

    def __len__(self):
        return len(self.x_train)


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size, padding=0, pad_if_needed=False):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            assert len(padding) == 2
            self.padding = padding

        self.pad_if_needed = pad_if_needed

    def __call__(self, image):

        ch = image.shape[2]
        iw = image.shape[0] + self.padding[0]
        ih = image.shape[1] + self.padding[1]

        image_tmp = np.zeros((iw, ih, ch))
        image_tmp[self.padding[0]:image.shape[0] + self.padding[0],
        self.padding[1]:image.shape[1] + self.padding[1], :] = image;

        h, w = image_tmp.shape[:2]
        new_h, new_w = self.output_size

        top = 0
        if h - new_h is not 0:
            top = np.random.randint(0, h - new_h)

        left = 0
        if w - new_w is not 0:
            left = np.random.randint(0, w - new_w)

        image = image_tmp[top: top + new_h,
                left: left + new_w, :]

        del image_tmp

        return image