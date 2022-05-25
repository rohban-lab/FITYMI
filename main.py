import argparse
import os
import numpy as np
import torch
from eval import evaluate_model
from models.modeling import CONFIGS, VisionTransformer
from train import train_model


def save_model(model, save_path):
    print('Saving model')
    torch.save(model.state_dict(), save_path)


def main(args):
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    print(f'Dataset: {args.dataset}, Normal Label: {args.label}')
    config = CONFIGS[args.backbone]
    model = VisionTransformer(config, args.vit_image_size, num_classes=2, zero_head=True)
    model.load_from(np.load(args.pretrained_path))
    model = model.to(device)
    model = train_model(args, model, device)
    save_model(model, os.path.join(args.output_dir, f'{args.backbone}_{args.dataset}_{args.label}.npy'))
    evaluate_model(args, model, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # General arguments
    parser.add_argument('--dataset', choices=['cifar10', 'cifar100'], default='cifar10',
                        help='The dataset used in the anomaly detection task')
    parser.add_argument('--epochs', default=30, type=int, help='The number of training epochs')
    parser.add_argument('--label', default=0, type=int, help='The normal class label')
    parser.add_argument('--learning_rate', type=float, default=4e-4, help='The initial learning rate of the optimizer')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='The weight decay of the optimizer')
    parser.add_argument('--train_batch_size', default=16, type=int)
    parser.add_argument('--eval_batch_size', default=16, type=int)
    parser.add_argument('--output_dir', default='results', type=str,
                        help='The directory used for saving the model results')
    parser.add_argument('--normal_data_path', default='data', type=str,
                        help='The path to the normal data')
    parser.add_argument('--gen_data_path', default='cifar10_training_gen_data.npy', type=str,
                        help='The path to the generated data')
    parser.add_argument('--download_dataset', action='store_true',
                        help="Whether to download datasets or not")
    parser.add_argument('--nnd', action='store_true',
                        help="Whether to evaluate on the NND setting or not")

    # Backbone arguments
    parser.add_argument('--backbone', choices=['ViT-B_16'], default='ViT-B_16', type=str, help='The ViT backbone type')
    parser.add_argument('--vit_image_size', default=224, type=int, help='The input image size of the ViT backbone')
    parser.add_argument('--pretrained_path', default='ViT-B_16.npz', type=str,
                        help='The path to the pretrained ViT weights')

    args = parser.parse_args()

    if args.dataset == 'cifar10':
        args.nnd_class_len = 100
    else:
        args.nnd = False

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args)
