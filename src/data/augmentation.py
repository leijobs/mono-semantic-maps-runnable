import torch
from torch.utils.data import Dataset

class AugmentedMapDataset(Dataset):

    def __init__(self, dataset, hflip=True):
        self.dataset = dataset
        self.hflip = hflip
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        image, pointcloud, calib, labels, mask = self.dataset[index]

        # Apply data augmentation
        if self.hflip:
            image, labels, mask = random_hflip(image, pointcloud, labels, mask)

        return image, pointcloud, calib, labels, mask

    
def random_hflip(image, pointcloud,  labels, mask):
    image = torch.flip(image, (-1,))
    pointcloud = torch.flip(pointcloud, (-1,))
    labels = torch.flip(labels.int(), (-1,)).bool()
    mask = torch.flip(mask.int(), (-1,)).bool()
    return image, pointcloud, labels, mask