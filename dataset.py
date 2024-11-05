import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader


class SkinDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        
        """
        Args:
            csv_file (string): Path to the CSV file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        
        img_name = self.data_frame.iloc[idx, 0] 
        labels = self.data_frame.iloc[idx, 1:].values.astype(float).argmax()
 
        img_path = os.path.join(self.root_dir, f"{img_name}.jpg")
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        labels = torch.tensor(labels, dtype=torch.float32)

        return image, labels


transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),        
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

dataset = SkinDataset(csv_file='dataset\GroundTruth.csv', root_dir='dataset\images', transform=transform)


dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

images, labels = next(iter(dataloader))
print(images.shape, labels.shape) 