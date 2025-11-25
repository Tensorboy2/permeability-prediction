'''

'''
from torch.utils.data import DataLoader, Dataset
import torch
torch.manual_seed(0)
import torchvision.transforms.functional as tf
import os
import multiprocessing
import numpy as np
path = os.path.dirname(__file__)
import random

class groupAugmentation:
    '''Data augmentation using 8 fixed image + permeability permutations (D4 group)'''
    def __init__(self):
        self.permutations = [
            ('I', lambda img: img, lambda K: K),
            ('R', lambda img: tf.rotate(img, 90),
                  lambda K: torch.tensor([K[3], -K[2], -K[1], K[0]])),  # 90°
            ('RR', lambda img: tf.rotate(img, 180),
                   lambda K: torch.tensor([K[0], K[1], K[2], K[3]])),  # 180°
            ('RRR', lambda img: tf.rotate(img, 270),
                    lambda K: torch.tensor([K[3], -K[2], -K[1], K[0]])),  # 270°
            ('H', lambda img: tf.hflip(img),
                  lambda K: torch.tensor([K[0], -K[1], -K[2], K[3]])),  # Horizontal flip
            ('HR', lambda img: tf.rotate(tf.hflip(img), 90),
                   lambda K: torch.tensor([K[3], K[2], K[1], K[0]])),  # H + 90°
            ('HRR', lambda img: tf.rotate(tf.hflip(img), 180),
                    lambda K: torch.tensor([K[0], -K[1], -K[2], K[3]])),  # H + 180°
            ('HRRR', lambda img: tf.rotate(tf.hflip(img), 270),
                     lambda K: torch.tensor([K[3], K[2], K[1], K[0]])),  # H + 270°
        ]

    def __call__(self, image, image_filled, label):
        K = label.clone()
        # Randomly choose one of the 8 permutations
        _, img_transform, k_transform = random.choice(self.permutations)

        image = img_transform(image)
        image_filled = img_transform(image_filled)
        K = k_transform(K)

        return image, image_filled, K

class DataAugmentation:
    '''Data agumentation class for flipping images and flipping label'''
    def __init__(self, hflip=True, vflip=True, rotate=True, p=0.5):
        self.hflip = hflip
        self.vflip = vflip
        self.rotate = rotate
        self.p = p
    
    def __call__(self, image, image_filled, label, *args, **kwds):
        K = label.clone()
        if self.hflip and random.random() < self.p:
            image = tf.hflip(image)
            image_filled = tf.hflip(image_filled)
            K[1] *= -1
            K[2] *= -1
            
        if self.vflip and random.random() < self.p:
            image = tf.vflip(image)
            image_filled = tf.vflip(image_filled)
            K[1] *= -1
            K[2] *= -1
        
        if self.rotate and random.random() < self.p:
            '''
            Rotation of image with corresponding permutation of K.
            '''
            angle = random.choice([90,180,270])
            image = tf.rotate(image, angle=angle)
            image_filled = tf.rotate(image_filled, angle=angle)

            Kxx, Kxy, Kyx, Kyy = K

            if angle == 90:
                K = torch.tensor([Kyy, -Kyx, -Kxy, Kxx])
            elif angle == 180:
                K = torch.tensor([Kxx, Kxy, Kyx, Kyy])
            elif angle == 270:
                K = torch.tensor([Kyy, -Kyx, -Kxy, Kxx])

        return image, image_filled, K
        

class CustomDataset(Dataset):
    '''
    Custom dataset for lazy loading with torch data_loader.
    Gives image, image_filled and label(2 by 2 permeability tensor.)
    Applies given transform using Data augmentation classes
    '''
    def __init__(self, label_path, image_path, image_filled_path, num_samples = None, transform = None):
        labels_all = np.load(label_path)
        images_all = np.load(image_path)
        images_filled_all = np.load(image_filled_path)

        # Limit by num_samples
        self.num_samples = num_samples or labels_all.shape[0]
        self.labels = labels_all[:self.num_samples]
        self.images = images_all[:self.num_samples]
        self.images_filled = images_filled_all[:self.num_samples]

        self.transform = transform
    
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = torch.from_numpy(self.images[idx]).float().unsqueeze(0)
        image_filled = torch.from_numpy(self.images_filled[idx]).float().unsqueeze(0)
        label = torch.from_numpy(self.labels[idx].flatten()).float()  # shape: (2, 2)

        if self.transform:
            image, image_filled, label = self.transform(image, image_filled, label)

        return image, image_filled, label

def get_available_workers():
    """
    Determine how many workers are truly available.
    This is especially important if slurm limits number of CPUs.
    """
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        return int(slurm_cpus)
    else:
        return multiprocessing.cpu_count()

def get_data(batch_size = 32,
             val_size=0.2, 
             hflip=True, 
             vflip=True, 
             rotate=True, 
             group=False,
             num_samples=None, 
             num_workers=None):
    '''
    Function for getting data and turning them into the train and test loader.
    Optional: normalization, grid search size, mask outliers.
    '''

    # Set up num workers for DataLoaders:
    available_cpus = get_available_workers()
    if num_workers is not None:
        if num_workers > available_cpus:
            print(f"Requested {num_workers} workers, but only {available_cpus} available. Using {available_cpus}.")
        num_workers = min(num_workers, available_cpus)
        print(f"Using {num_workers} available workers.")
    else:
        num_workers = available_cpus
        print(f"Using {num_workers} available workers.")

    # Paths:
    image_path = os.path.join(path, 'data/images.npy')
    image_filled_path = os.path.join(path, 'data/images_filled.npy')
    label_path = os.path.join(path, 'data/train_and_val_k.npy')


    if group:
        aug = groupAugmentation()
    else:
        aug = DataAugmentation(hflip=hflip, 
                               vflip=vflip,
                               rotate=rotate,
                               p=0.5)


    dataset = CustomDataset(
        image_path=image_path,
        image_filled_path=image_filled_path,
        label_path=label_path,
        num_samples=num_samples,
        transform=aug,
    )


    # Split into train and test sets with fixed seed for reproducibility:
    num_samples = len(dataset)
    print(f"Num datapoints: {num_samples}")
    train_size = int((1 - val_size) * num_samples)
    val_size = num_samples - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)

    # Create DataLoaders for train and test datasets with fixed seed for shuffling:
    train_data_loader = DataLoader(train_dataset, 
                                    batch_size=batch_size, 
                                    shuffle=True, 
                                    num_workers=num_workers,
                                    pin_memory=True,
                                    generator=generator)

    val_data_loader = DataLoader(val_dataset, 
                                    batch_size=batch_size, 
                                    shuffle=False, 
                                    num_workers=num_workers,
                                    pin_memory=True)

    return train_data_loader, val_data_loader


def print_dataset_size_gb():
    image_path = os.path.join(path, 'data/images.npy')
    image_filled_path = os.path.join(path, 'data/images_filled.npy')
    label_path = os.path.join(path, 'data/train_and_val_k.npy')
    total_bytes = 0
    for f in [image_path, image_filled_path, label_path]:
        if os.path.exists(f):
            total_bytes += os.path.getsize(f)
        else:
            print(f"Warning: {f} does not exist.")
    print(f"Total dataset size: {total_bytes / (1024**3):.2f} GB")

if __name__ == "__main__":
    print_dataset_size_gb()
    print("Getting data...")
    train_data_loader, test_data_loader = get_data(batch_size=128,num_workers=2,num_samples=None)
    print(f"Len train loader: {len(train_data_loader)}")
    print(f"Len test loader: {len(test_data_loader)}")
    print("Data loaders ready, with lazy loading.")
