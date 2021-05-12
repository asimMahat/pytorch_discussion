import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io

class catDogDataset(Dataset):
    
    def __init__(self,csv_file,root_dir,transform = False):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__ (self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir,self.annotations.iloc[index,0])
        img = io.imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))
        
        if self.transform:
            img = self.transform(img)
            img = transform.Resize(img)
        
        return(img,y_label)
    
        