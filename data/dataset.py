from torch.utils.data import Dataset
from torchvision.io import read_image
import os
from torchvision.transforms import Resize
from torchvision.transforms.functional import rotate
import torch

class WaterMeterSegDatset(Dataset):
    def __init__(self, images_dir:str, masks_dir:str, transform: callable=None, input_size:tuple=(500, 500), mask_size=(504, 504)) -> None:
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

        self.image_resize = Resize(input_size)
        self.mask_resize = Resize(mask_size)
    def __len__(self):
        return len(self.names)

    def __getitem__(self, index: int):
        # read images
        image_path = os.path.join(self.images_dir, self.names[index])
        mask_path = os.path.join(self.masks_dir, self.names[index])
        image, mask = read_image(image_path).float(), read_image(mask_path).float()

        # fix dataset bug
        if image.shape[1] != mask.shape[1]:
            image = image.permute(0, 2, 1).flip(2)
        
        image, mask = self.image_resize(image), self.mask_resize(mask)
        mask = mask / 255

        # apply transforms
        if self.transform:
            stack = torch.cat(image, mask, dim=0)
            image, mask = self.transform(stack)
            image, mask = torch.chunk(stack, dim=0)
        
        return image, mask