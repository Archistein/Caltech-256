import torch
from torch.utils import data
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from collections import Counter
from typing import Optional
import cv2


class Caltech256(data.Dataset):
    def __init__(self, file_path: str, transforms: A.Compose) -> None:
        with open(file_path, 'r') as f:
            self.images_paths = f.readlines()
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.images_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        image_path, target, _ = self.images_paths[idx].split()
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)['image']

        return image, int(target)
    
    def get_transforms() -> A.Compose:
        return {
            'train': A.Compose([
                A.Resize(width=256, height=256),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=40, p=0.6),
                A.RandomCrop(width=224, height=224),
                A.RandomBrightnessContrast(p=0.6),
                A.AdvancedBlur(p=0.6),
                A.HorizontalFlip(p=0.5),
                A.GridDistortion(num_steps=3, p=0.6),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]), 
            'val': A.Compose([
                A.Resize(width=256, height=256),
                A.CenterCrop(width=224, height=224),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]) 
        }


def get_dataloaders(split_root: str, batch_size: int) -> dict[str, data.DataLoader]:
    transforms = Caltech256.get_transforms()

    with open(f'{split_root}/train_lst.txt', 'r') as f:
        targets = list(map(lambda x: int(x.split()[1]), f.readlines()))
        freq_map = Counter(targets)
        sample_weights = list(map(lambda x: 1/freq_map[x], targets))
        
    sampler = data.WeightedRandomSampler(sample_weights, num_samples = len(sample_weights), replacement=True)

    datasets = {x: Caltech256(f'{split_root}/{x}_lst.txt', transforms[x]) for x in ['train', 'val']}
    dataloaders = {
                    'train': data.DataLoader(datasets['train'], batch_size=batch_size, sampler=sampler, num_workers=4),
                    'val': data.DataLoader(datasets['val'], batch_size=batch_size, num_workers=4)
    }

    return dataloaders


def visualize_batch(batch: tuple[torch.Tensor, torch.Tensor], 
                    grid_size: int,
                    labels: list[str], 
                    mean: Optional[torch.Tensor] = None,
                    std: Optional[torch.Tensor] = None) -> None:
    _, axs = plt.subplots(nrows=grid_size, ncols=grid_size, figsize=(20, 20))

    inps, targets = batch

    if mean is None:
        mean = torch.tensor([0.485, 0.456, 0.406])
    if std is None:
        std = torch.tensor([0.229, 0.224, 0.225])

    for i in range(grid_size):
        for j in range(grid_size):
            axs[i][j].set_title(labels[targets[i * grid_size + j]])
            inp = inps[i * grid_size + j].permute(1, 2, 0)
            inp = std * inp + mean
            axs[i][j].imshow(inp)
            axs[i][j].get_xaxis().set_ticks([])
            axs[i][j].get_yaxis().set_ticks([])
            
    # plt.tight_layout()
    plt.show()