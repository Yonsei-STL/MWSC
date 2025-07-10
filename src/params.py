import argparse
from pprint import pprint

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_path', default=None, help="image data path")
    parser.add_argument('--train_label_path', default=None, help="train label csv file path")
    parser.add_argument('--val_label_path', default=None, help="validation label csv file path")
    parser.add_argument('--output_path', default=None, help="state dict data save path")
    parser.add_argument('--pretrain', type=str, default='False', help="timm model pretrain")
    parser.add_argument('--clip_base_model', type=str, default='ViT-B/32', choices=['ViT-B/32', 'ViT-B/16', 'ViT-L/14'], help="backbone for clip")
    parser.add_argument('--timm_model_name', type=str, help="timm of the model to be fine-tuned")
    parser.add_argument('--ablation_mode', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--num_workers', type=int, default=8)

    args = parser.parse_args()
    print('Called with args:')
    pprint(vars(args), indent=2)

    return args