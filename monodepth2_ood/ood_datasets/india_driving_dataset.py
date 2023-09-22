import os
import os.path
from PIL import Image
import numpy as np
import torch
import torch.utils.data as data
from id_datasets import transforms
from natsort import natsorted

iheight, iwidth = 256, 256 # raw image size


def rgb_loader(path):
    img = Image.open(path, "r")
    np_img = np.array(img)
    return np_img


class MyDataloader(data.Dataset):

    def is_image_file(self, filename):
        IMG_EXTENSIONS = ['.png']
        return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

    def make_dataset(self, dir, num_imgs):
        images = []
        counter = 0
        dir = os.path.expanduser(dir)
        for subdir in natsorted(os.listdir(dir)):
            d = os.path.join(dir, subdir)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if counter < num_imgs and self.is_image_file(fname):
                        path = os.path.join(root, fname)
                        images.append(path)
                        counter += 1
        return images

    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4)

    def __init__(self, root, loader=rgb_loader):
        """
        Note: we do not use a test split, since we do only use the dataset as test data
        """
        self.loader = loader
        root_path = os.getcwd()
        self.root = os.path.join(root_path, root, 'leftImg8bit', 'test')
        self.transform = self.val_transform
        self.num_imgs = 300
        imgs = self.make_dataset(self.root, self.num_imgs)
        assert len(imgs) > 0, "Found 0 images in subfolders of: " + self.root + "\n"
        self.imgs = imgs
        self.len = len(self.imgs)

    def val_transform(self, rgb):
        raise (RuntimeError("val_transform is not implemented."))

    def __getraw__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (rgb, depth) the raw data.
        """
        path = self.imgs[index]
        rgb = self.loader(path)
        return rgb

    def __getitem__(self, index):
        rgb = self.__getraw__(index)
        if self.transform is not None:
            rgb_np = self.transform(rgb)
        else:
            raise(RuntimeError("transform not defined"))

        to_tensor = transforms.ToTensor()
        input_tensor = to_tensor(rgb_np)
        while input_tensor.dim() < 3:
            input_tensor = input_tensor.unsqueeze(0)

        # also return placeholder for depth
        # return input_tensor, torch.zeros(input_tensor.size())
        inputs = {}
        inputs[("color", 0, 0)] = input_tensor
        inputs["depth_gt"] = torch.zeros(input_tensor.size())
        return inputs

    def __len__(self):
        return self.len


class IndiaDrivingDataset(MyDataloader):
    def __init__(self, root, height=192, width=640):
        super(IndiaDrivingDataset, self).__init__(root)
        self.output_size = (height, width)

    def is_image_file(self, filename):
        return filename.endswith('.png')

    def val_transform(self, rgb):
        transform = transforms.Compose([
            transforms.Resize(self.output_size),
        ])

        rgb_np = transform(rgb)

        rgb_np = np.asfarray(rgb_np, dtype='float') / 255
        return rgb_np
