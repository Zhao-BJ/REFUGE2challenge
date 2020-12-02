import imageio
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image


class Iset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.transform = transform
        df = pd.read_csv(data_dir+"label.csv")
        self.img_name = df["img"].values
        self.img = [imageio.imread(data_dir+"img/"+i) for i in self.img_name]

    def __getitem__(self, index):
        img, name = self.img[index], self.img_name[index]
        img = Image.fromarray(img, mode='RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, name

    def __len__(self):
        return len(self.img_name)
