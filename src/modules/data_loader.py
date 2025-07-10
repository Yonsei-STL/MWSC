import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import ast
import os

class WeatherDataset(Dataset):
    def __init__(self, csv_file, transform=None, data_dir='/.data/'):
        self.data = pd.read_csv(csv_file, sep='|')
        self.transform = transform
        self.data_dir = data_dir

        self.data['filepath'] = self.data['filepath'].apply(self.modify_path)
        
        self.weather_types = ['clear', 'foggy', 'snowy', 'rainy']
        self.severity_levels = ['light', 'moderate', 'heavy']
        
        self.weather_to_idx = {w: i for i, w in enumerate(self.weather_types)}
        self.severity_to_idx = {s: i for i, s in enumerate(self.severity_levels)}
        
        self.prompts = self.generate_prompt()

    def generate_prompt(self):
        weather_prompt = []
        severity_prompt = []
        for r in self.weather_types:
            p = f"A photo of weather {r}."
            weather_prompt.append(p)
        for r in self.severity_levels:
            p = f"A photo of weather severity {r}."
            severity_prompt.append(p)                    
        
        prompt = weather_prompt + severity_prompt

        return prompt

    def modify_path(self, path):
        parts = path.split('/')
        subfolder = '/'.join(parts[-3:])
        
        new_path = os.path.join(self.data_dir, subfolder)
        return new_path

    def get_weather_condition(self, folder_name):
        weather_mapping = {
            'clear': 'clear',
            'fog': 'foggy',
            'snow': 'snowy',
            'rain': 'rainy'
        }
        return weather_mapping.get(folder_name, folder_name)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['filepath']
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"File not found: {img_path}")
            return None

        if self.transform:
            image = self.transform(image)

        weather_list = ast.literal_eval(self.data.iloc[idx]['weather'])
        weather_label = torch.zeros(len(self.weather_types))
        for w in weather_list:
            weather_label[self.weather_to_idx[w.lower()]] = 1

        severity = self.data.iloc[idx]['severity'].lower()
        severity = severity.split('\'')[1]
        severity_label = torch.tensor(self.severity_to_idx[severity])

        return image, weather_label, severity_label