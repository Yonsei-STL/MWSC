import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import timm

import numpy as np
from tqdm import tqdm

from modules.data_loader import WeatherDataset
from modules.classifier_finetune import Finetune_model
from modules.metric import Metric
from params import parse_args

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.pretrain == 'True':
        m = timm.create_model(args.timm_model_name, pretrained=True)
    else:
        m = timm.create_model(args.timm_model_name, pretrained=False)
    data_config = timm.data.resolve_model_data_config(m)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    train_dataset = WeatherDataset(args.train_label_path, transform=transforms, data_dir=args.image_path)
    val_dataset = WeatherDataset(args.val_label_path, transform=transforms, data_dir=args.image_path)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    weather_types = train_dataset.weather_types
    severity_levels = train_dataset.severity_levels

    weather_criterion = nn.BCEWithLogitsLoss()
    severity_criterion = nn.CrossEntropyLoss()
    
    metric = Metric(weather_types, severity_levels)

    model = Finetune_model(args.timm_model_name, len(train_dataset.weather_types), len(train_dataset.severity_levels), args.pretrain)
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0.0
        for images, weather_labels, severity_labels in tqdm(train_loader):
            images = images.to(device)
            weather_labels = weather_labels.to(device)
            severity_labels = severity_labels.to(device)
            
            optimizer.zero_grad()
            weather_out, severity_out = model(images)

            weather_loss = weather_criterion(weather_out, weather_labels)
            severity_loss = severity_criterion(severity_out, severity_labels)
            loss = weather_loss + severity_loss
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        all_weather_probs = []
        all_weather_labels = []
        all_severity_probs = []
        all_severity_labels = []
        
        with torch.no_grad():
            for images, weather_labels, severity_labels in tqdm(val_loader):
                images = images.to(device)
                weather_labels = weather_labels.to(device)
                severity_labels = severity_labels.to(device)
                
                weather_out, severity_out = model(images)
                
                weather_loss = weather_criterion(weather_out, weather_labels)
                severity_loss = severity_criterion(severity_out, severity_labels)
                loss = weather_loss + severity_loss
                val_loss += loss.item()

                weather_probs = torch.sigmoid(weather_out).cpu().numpy()
                all_weather_probs.extend(weather_probs)
                all_weather_labels.extend(weather_labels.cpu().numpy())

                severity_probs = F.softmax(severity_out, dim=1).cpu().numpy()
                all_severity_probs.extend(severity_probs)
                all_severity_labels.extend(severity_labels.cpu().numpy())

        weather_metrics = metric.calculate(np.array(all_weather_probs), np.array(all_weather_labels), is_multilabel=True)
        severity_metrics = metric.calculate(np.array(all_severity_probs), np.array(all_severity_labels), is_multilabel=False)

        print(f'Epoch {epoch+1}/{args.num_epochs}')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}')
        
        print(f'Weather Accuracy: {weather_metrics["accuracy"]:.4f}')
        print('\nWeather Classification Report:')
        print(weather_metrics['classification_report'])
        print('\nWeather Confusion Matrix:')
        print(weather_metrics['confusion_matrix'])
        print('\nSeverity Classification Report:')
        print(severity_metrics['classification_report'])
        print('-' * 50)
    if args.pretrain == 'True':
        torch.save(model.state_dict(), f'{args.output_path}finetune_{args.timm_model_name}_pretrain.pth')
    else:
        torch.save(model.state_dict(), f'{args.output_path}finetune_{args.timm_model_name}.pth')
    
if __name__ == "__main__":
    args = parse_args()
    main(args)