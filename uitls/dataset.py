from torch.utils.data import Dataset
from torchvision.io import read_image
import os

class WaterMeterSegDatset(Dataset):
    def __init__(self, images_dir:str, masks_dir:str, transform: callable=None) -> None:
        '''
        Initialize WaterMeterSeqDataset

        Params:
            images_dir(str): path to images
            masks_dir(str): path to masks
            trasnsform(callable, optional): transform for source and mask
        '''
        super().__init__()
        self.names = sorted(os.listdir(images_dir))

        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index: int):
        # read images
        image_path = os.path.join(self.images_dir, self.names[index])
        mask_path = os.path.join(self.masks_dir, self.names[index])
        image, mask = read_image(image_path), read_image(mask_path)
        
        # apply transforms
        if self.transform:
            image, mask = self.transform(image), self.transform(mask)
        
        return image, mask