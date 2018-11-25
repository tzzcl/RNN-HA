import torch
import scipy.io as sio
from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader
# ======================= define model ===================================
class CarDataset(Dataset):
    """Car dataset."""

    def __init__(self, mat_file, root_dir, transform=None):

        self.mat_file = sio.loadmat(mat_file)
        self.name = self.mat_file['images']['name'][0][0][0]
        self.model = self.mat_file['images']['model'][0][0][0]
        self.label = self.mat_file['images']['class'][0][0][0]
        self.root_dir = root_dir
        self.transform = transform
    def __len__(self):
        return (self.label.shape[0])

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.name[idx][0])
        image = Image.open(img_name).convert('RGB')
        model = (self.model[idx] - 1).astype('int64')
        label = (self.label[idx] - 1).astype('int64')
        sample = {'image': image, 'model':model,'label':label}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample
class Self_Scale(object):
    """Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        return img.resize((self.size, self.size), self.interpolation)
class Invert_Normalize(object):
    def __init__(self, mean = None):
        self.mean = [123.68,116.78,103.94]
        self.s = 1. / 255
    def __call__(self, tensor):
        # TODO: make efficient
        for t, m in zip(tensor, self.mean):
            t.div(self.s).sub_(m)
        return tensor