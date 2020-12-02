import imageio
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class IMset(Dataset):
    def __init__(self, data_dir, transform=None, img='img', mask='mask', mask_mode='RGB'):
        self.transform = transform
        self.mask_mode = mask_mode
        df = pd.read_csv(data_dir + "label.csv")
        self.img_name = df["img"].values
        self.img = [imageio.imread(data_dir + img + "/" + i) for i in self.img_name]
        self.mask = [imageio.imread(data_dir + mask + "/" + i) for i in self.img_name]

    def __getitem__(self, index):
        img, mask, name = self.img[index], self.mask[index], self.img_name[index]
        img = Image.fromarray(img, mode='RGB')
        mask = Image.fromarray(mask, mode=self.mask_mode)
        if self.transform is not None:
            img = self.transform(img)
            mask = self.transform(mask)
        return img, mask, name

    def __len__(self):
        return len(self.img_name)
