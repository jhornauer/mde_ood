import os
import os.path
from PIL import Image
import numpy as np
import torch
import torch.utils.data as data
from id_datasets import transforms

iheight, iwidth = 256, 256 # raw image size


def rgb_loader(path):
    img = Image.open(path, "r")
    np_img = np.array(img)
    return np_img


class MyDataloader(data.Dataset):

    def is_image_file(self, filename):
        IMG_EXTENSIONS = ['.jpg']
        return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

    def make_dataset(self, dir, categories, imgs_per_category=50):
        images = []
        dir = os.path.expanduser(dir)
        for subdir in sorted(os.listdir(dir)):
            d = os.path.join(dir, subdir)
            if not os.path.isdir(d):
                continue
            for target in sorted(os.listdir(d)):
                dt = os.path.join(d, target)
                if target not in categories or not os.path.isdir(dt):
                    continue
                for root, _, fnames in sorted(os.walk(dt)):
                    counter = 0
                    for fname in sorted(fnames):
                        if counter < imgs_per_category and self.is_image_file(fname):
                            path = os.path.join(root, fname)
                            images.append(path)
                            counter += 1
        return images

    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4)

    def __init__(self, root, loader=rgb_loader, scenes='indoor'):
        """
        Note: we do not use a test split, since we do only use the dataset as test data
        """
        self.loader = loader
        root_path = os.getcwd()
        self.root = os.path.join(root_path, root, 'data_256')
        if scenes == 'indoor':
            self.categories = ['butte', 'cliff', 'corn_field', 'desert_road', 'harbor', 'highway']
            self.transform = self.val_transform_indoor
        elif scenes == 'outdoor':
            self.categories = ['art_gallery', 'bathroom', 'dining_room', 'home_office', 'hospital_room', 'kitchen']
            self.transform = self.val_transform_outdoor
        else:
            raise RuntimeError("scene type not implemented")
        self.imgs_per_category = 50
        imgs = self.make_dataset(self.root, self.categories, self.imgs_per_category)
        assert len(imgs) > 0, "Found 0 images in subfolders of: " + self.root + "\n"
        self.imgs = imgs
        self.len = len(self.imgs)

    def val_transform_indoor(self, rgb):
        raise (RuntimeError("val_transform_indoor() is not implemented."))

    def val_transform_outdoor(self, rgb):
        raise (RuntimeError("val_transform_outdoor() is not implemented."))

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


class Places365Dataset(MyDataloader):
    def __init__(self, root, height=224, width=288, scenes='indoor'):
        super(Places365Dataset, self).__init__(root, scenes=scenes)
        self.output_size = (height, width)

    def is_image_file(self, filename):
        return filename.endswith('.jpg')

    def val_transform_indoor(self, rgb):
        transform = transforms.Compose([
            transforms.Resize(288.0 / iheight),
            transforms.CenterCrop(self.output_size),
        ])

        rgb_np = transform(rgb)

        rgb_np = np.asfarray(rgb_np, dtype='float') / 255
        return rgb_np

    def val_transform_outdoor(self, rgb):
        transform = transforms.Compose([
            transforms.Resize(self.output_size),
        ])

        rgb_np = transform(rgb)

        rgb_np = np.asfarray(rgb_np, dtype='float') / 255
        return rgb_np
