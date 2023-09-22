import os
import os.path
from PIL import Image
import torch
import torch.utils.data as data
from torchvision import transforms

iheight, iwidth = 256, 256 # raw image size


def rgb_loader(path):
    img = Image.open(path, "r")
    return img


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

    def __init__(self, root, loader=rgb_loader):
        """
        Note: we do not use a test split, since we do only use the dataset as test data
        """
        self.loader = loader
        root_path = os.getcwd()
        self.root = os.path.join(root_path, root, 'data_256')
        self.categories = ['art_gallery', 'bathroom', 'dining_room', 'home_office', 'hospital_room', 'kitchen']
        self.transform = self.val_transform_outdoor
        self.imgs_per_category = 50

        imgs = self.make_dataset(self.root, self.categories, self.imgs_per_category)
        assert len(imgs) > 0, "Found 0 images in subfolders of: " + self.root + "\n"
        self.imgs = imgs
        self.len = len(self.imgs)

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
            input_tensor = self.transform(rgb)
        else:
            raise(RuntimeError("transform not defined"))

        to_tensor = transforms.ToTensor()
        input_tensor = to_tensor(input_tensor)
        while input_tensor.dim() < 3:
            input_tensor = input_tensor.unsqueeze(0)

        # also return placeholder for depth
        inputs = {}
        inputs[("color", 0, 0)] = input_tensor
        inputs["depth_gt"] = torch.zeros(input_tensor.size())
        return inputs

    def __len__(self):
        return self.len


class Places365Dataset(MyDataloader):
    def __init__(self, root, height, width):
        super(Places365Dataset, self).__init__(root)
        self.output_size = (height, width)

    def is_image_file(self, filename):
        return filename.endswith('.jpg')

    def val_transform_outdoor(self, rgb):
        transform = transforms.Compose([transforms.Resize(self.output_size)])

        rgb = transform(rgb)
        return rgb
