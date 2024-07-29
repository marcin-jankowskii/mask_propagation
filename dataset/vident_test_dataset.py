import os
from os import path
import numpy as np
from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data.dataset import Dataset

from dataset.range_transform import im_normalization  # importować zgodnie z rzeczywistą lokalizacją
from dataset.util import all_to_onehot  # importować zgodnie z rzeczywistą lokalizacją


class VidentDataset(Dataset):
    def __init__(self, root, split='train', resolution='480p', target_name=None):
        self.root = root
        self.split = split
        self.resolution = resolution

        self.data_dir = path.join(root, split)
        self.sequences = [d for d in os.listdir(self.data_dir) if path.isdir(path.join(self.data_dir, d))]
        if target_name:
            self.sequences = [s for s in self.sequences if s == target_name]

        self.num_frames = {}
        self.num_objects = {}
        self.shape = {}
        for seq in self.sequences:
            gt_dir = path.join(self.data_dir, seq, 'GT')
            mask_dir = path.join(self.data_dir, seq, 'new_masks', 'mask')
            self.num_frames[seq] = len([name for name in os.listdir(gt_dir) if name.endswith('.png')])
            _mask = np.array(Image.open(path.join(mask_dir, '00001.png')).convert("P"))
            self.num_objects[seq] = np.max(_mask)
            self.shape[seq] = np.shape(_mask)

        self.im_transform = transforms.Compose([
            #transforms.Resize((200, 200)),  # Resize to 400x400
            transforms.ToTensor(),
            im_normalization,
        ])

        self.mask_transform = transforms.Compose([
            #transforms.Resize((200, 200), interpolation=Image.NEAREST),
        ])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence = self.sequences[index]
        info = {}
        info['name'] = sequence
        info['num_frames'] = self.num_frames[sequence]

        images = []
        masks = []
        for f in range(1, self.num_frames[sequence] + 1):
            img_file = path.join(self.data_dir, sequence, 'GT', '{:05d}.png'.format(f))
            images.append(self.im_transform(Image.open(img_file).convert('RGB')))

            mask_file = path.join(self.data_dir, sequence, 'new_masks', 'mask', '{:05d}.png'.format(f))
            if path.exists(mask_file):
                mask = Image.open(mask_file).convert('P')
                mask = self.mask_transform(mask)
                masks.append(np.array(mask, dtype=np.uint8))
            else:
                masks.append(np.zeros((800, 800), dtype=np.uint8))

        images = torch.stack(images, 0)
        masks = np.stack(masks, 0)

        labels = np.unique(masks)
        labels = labels[labels != 0]  # Ignorowanie tła oznaczonego jako 0
        masks = torch.from_numpy(all_to_onehot(masks, labels)).float()

        masks = masks.unsqueeze(2)

        info['labels'] = labels

        data = {
            'rgb': images,
            'gt': masks,
            'info': info,
        }

        return data

# Przykład użycia:
# from torch.utils.data import DataLoader

# dataset = CustomDataset(root='path/to/dataset', split='train')
# dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# for data in dataloader:
#     images = data['rgb']
#     masks = data['gt']
#     info = data['info']
#     # Przetwarzaj dane dalej...
