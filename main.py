from train_eval import train, eval
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
from dataset import BabyBeatDataset
import os
import torch
import random
import numpy as np
import argparse
from select_model import select_model
from typing import Literal

if __name__ == "__main__":
    # 设置当前工作目录为脚本所在的目录
    current_script_path = os.path.abspath(__file__)
    current_script_directory = os.path.dirname(current_script_path)
    os.chdir(current_script_directory)

    # 设置随机种子
    fix_seed = 2024
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser()
    # basic config
    parser.add_argument(
        "--model", type=str, required=True, default="MyNet_4", help="model name"
    )
    # gpu
    parser.add_argument("--use_gpu", type=bool, default=True, help="use gpu or not")
    parser.add_argument("--gpu", type=str, default="0", help="gpu id")
    # train
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--num_epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--patience", type=int, default=15, help="patience")
    # model
    parser.add_argument("--num_classes", type=int, default=2, help="number of classes")
    parser.add_argument("--in_channels", type=int, default=1, help="number of channels")
    parser.add_argument("--seq_len", type=int, default=4800, help="sequence length")
    parser.add_argument(
        "--input_feature",
        type=str,
        default="fhr",
        help="Input feature (fhr, ucp, or both)",
    )
    parser.add_argument("--kernel_size", type=int, default=3, help="kernel size")
    args = parser.parse_args()

    # 2. select gpu
    if args.use_gpu:
        gpu = "cuda:" + args.gpu
        device = torch.device(gpu if torch.cuda.is_available() else "cpu")
        print(">>> use ", device)

    # 3. select model
    model = select_model(args, device)
    # model_class_name = model.__class__.__name__
    # model_save_name = f"{model_class_name}.pth"
    model_save_name = f"{args.model}_{args.batch_size}bs_{args.num_epochs}epoc_{args.kernel_size}ks_{args.input_feature}.pth"
    print(">>> model: ", model_save_name)

    # 1. get dataloader
    path = r"./dataset/BabyBeatAnalyzer.ts"
    dataset = BabyBeatDataset(path)
    # 划分训练集和测试集
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    _train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    # 从训练集中划分出一部分作为验证集
    train_size = int(0.9 * len(_train_dataset))
    val_size = len(_train_dataset) - train_size
    train_dataset, val_dataset = random_split(_train_dataset, [train_size, val_size])

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    all_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        patience=args.patience,
        device=device,
        model_save_name=model_save_name,
        input_feature=args.input_feature,
    )

    eval(
        model=model,
        val_loader=test_loader,
        device=device,
        input_feature=args.input_feature,
    )
