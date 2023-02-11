from torchvision.transforms.functional import affine, vflip, hflip
import random

class RandomAffine(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        image, mask = sample

        angle = random.randint(-180, 180)  
        translate = [random.randint(30) for _ in range(2)]
        shear = random.randint(-180, 180)

        image, mask = affine(image, angle, translate, shear), affine(mask, angle, translate, shear)

        return image, mask

class RandomFlip(object):
    def __init__(self) -> None:
        pass

    def __call__(self, sample):
        
        image, mask = sample
        
        if random.randint(0, 1) == 1:
            image, mask = vflip(image), vflip(mask)

        if random.randint(0, 1):
            image, mask = hflip(image), hflip(mask)

        return image, mask


