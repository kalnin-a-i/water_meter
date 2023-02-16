from torchvision.transforms.functional import affine, vflip, hflip
import random

class RandomAffine(object):
    def __init__(self, max_angle=180, max_translate=40, min_scale=0.5, max_scale=2, max_shear=180):
        self.max_angle = max_angle
        self.max_translate= max_translate
        self.max_scale = max_scale
        self.min_scale = min_scale
        self.max_shear = max_shear

        if self.min_scale >= self.max_scale:
            raise Exception('min_scale >= max_scale')
        pass

    def __call__(self, sample):
        image, mask = sample

        angle = random.randint(-self.max_angle, self.max_angle)  
        translate = [random.randint(0, self.max_translate) for _ in range(2)]
        shear = random.randint(-self.max_shear, self.max_shear)
        scale = self.min_scale + random.random() * (self.max_scale - self.min_scale)

        image, mask = affine(image, angle, translate, scale, shear), affine(mask, angle, translate, scale, shear)

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


