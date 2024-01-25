import argparse
import os
import torch
import random
import numpy as np
from train_eval_transformer import train, eval
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
    fix_seed = 2024
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description="TimesNet")

    # basic config
    parser.add_argument(
        "--task_name",
        type=str,
        # required=True,
        default="classification",
        help="task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]",
    )
    # parser.add_argument(
    #     "--is_training", type=int, required=True, default=1, help="status"
    # )
    # parser.add_argument(
    #     "--model_id", type=str, required=True, default="test", help="model id"
    # )
    parser.add_argument(
        "--model",
        type=str,
        default="TimesNet",
        help="model name, options: [Autoformer, Transformer, TimesNet]",
    )

    parser.add_argument(
        "--root_path",
        type=str,
        default="./data/ETT/",
        help="root path of the data file",
    )
    parser.add_argument("--data_path", type=str, default="ETTh1.csv", help="data file")
    parser.add_argument(
        "--features",
        type=str,
        default="M",
        help="forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate",
    )
    parser.add_argument(
        "--target", type=str, default="OT", help="target feature in S or MS task"
    )
    parser.add_argument(
        "--freq",
        type=str,
        default="h",
        help="freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h",
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        default="./checkpoints/",
        help="location of model checkpoints",
    )

    # forecasting task
    parser.add_argument(
        "--seq_len", type=int, default=4800, help="input sequence length"
    )
    parser.add_argument("--label_len", type=int, default=48, help="start token length")
    parser.add_argument(
        "--pred_len", type=int, default=0, help="prediction sequence length"
    )
    parser.add_argument(
        "--seasonal_patterns", type=str, default="Monthly", help="subset for M4"
    )

    # inputation task
    parser.add_argument("--mask_rate", type=float, default=0.25, help="mask ratio")

    # anomaly detection task
    parser.add_argument(
        "--anomaly_ratio", type=float, default=0.25, help="prior anomaly ratio (%)"
    )

    # model define
    parser.add_argument("--num_class", type=int, default=2, help="number of classes")
    parser.add_argument("--top_k", type=int, default=1, help="for TimesBlock")
    parser.add_argument("--num_kernels", type=int, default=6, help="for Inception")
    parser.add_argument("--enc_in", type=int, default=2, help="encoder input size")
    parser.add_argument("--dec_in", type=int, default=7, help="decoder input size")
    parser.add_argument("--c_out", type=int, default=7, help="output size")
    parser.add_argument("--d_model", type=int, default=64, help="dimension of model")
    parser.add_argument("--n_heads", type=int, default=8, help="num of heads")
    parser.add_argument("--e_layers", type=int, default=2, help="num of encoder layers")
    parser.add_argument("--d_layers", type=int, default=1, help="num of decoder layers")
    parser.add_argument("--d_ff", type=int, default=2048, help="dimension of fcn")
    parser.add_argument(
        "--moving_avg", type=int, default=25, help="window size of moving average"
    )
    parser.add_argument("--factor", type=int, default=1, help="attn factor")
    parser.add_argument(
        "--distil",
        action="store_false",
        help="whether to use distilling in encoder, using this argument means not using distilling",
        default=True,
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
    parser.add_argument(
        "--embed",
        type=str,
        default="timeF",
        help="time features encoding, options:[timeF, fixed, learned]",
    )
    parser.add_argument("--activation", type=str, default="gelu", help="activation")
    parser.add_argument(
        "--output_attention",
        action="store_true",
        help="whether to output attention in ecoder",
    )
    parser.add_argument(
        "--input_feature",
        type=str,
        default="fhr",
        help="Input feature (fhr, ucp, or both)",
    )

    # optimization
    parser.add_argument(
        "--num_workers", type=int, default=10, help="data loader num workers"
    )
    parser.add_argument("--itr", type=int, default=1, help="experiments times")
    parser.add_argument("--num_epochs", type=int, default=100, help="num_epochs")
    parser.add_argument(
        "--batch_size", type=int, default=8, help="batch size of train input data"
    )
    parser.add_argument(
        "--patience", type=int, default=15, help="early stopping patience"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.0001, help="optimizer learning rate"
    )
    parser.add_argument("--des", type=str, default="test", help="exp description")
    parser.add_argument("--loss", type=str, default="MSE", help="loss function")
    parser.add_argument(
        "--lradj", type=str, default="type1", help="adjust learning rate"
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="use automatic mixed precision training",
        default=False,
    )

    # gpu
    parser.add_argument("--use_gpu", type=bool, default=True, help="use gpu or not")
    parser.add_argument("--gpu", type=str, default="0", help="gpu id")

    # de-stationary projector params
    parser.add_argument(
        "--p_hidden_dims",
        type=int,
        nargs="+",
        default=[128, 128],
        help="hidden layer dimensions of projector (List)",
    )
    parser.add_argument(
        "--p_hidden_layers",
        type=int,
        default=2,
        help="number of hidden layers in projector",
    )

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
    model_save_name = f"{args.model}_{args.batch_size}bs_{args.num_epochs}epoc_{args.input_feature}.pth"
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
    )

    eval(model=model, val_loader=test_loader, device=device)

    # model = Model(args)
    # # 创建测试输入
    # test_input = torch.randn(16, 4800, 2)  # 随机生成符合输入形状的数据
    # # 全部是1
    # test_mark_input = torch.ones(16, 4800)
    # # 进行预测
    # output = model(test_input, test_mark_input)

    # # 检查输出形状
    # print("Output Shape:", output.shape)
