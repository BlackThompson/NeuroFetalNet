# BiGRU.py   NeuroFetalNet.py                   NeuroFetalNet_without_pe.py
# BiLSTM.py  NeuroFetalNet_without_ca.py        ResNet_BiGRU.py
# GRU.py     NeuroFetalNet_without_fusion_3.py  ResNet.py
# LSTM.py    NeuroFetalNet_without_fusion_9.py

from ablation_model.BiGRU import BiGRU
from ablation_model.BiLSTM import BiLSTM
from ablation_model.NeuroFetalNet import NeuroFetalNet
from ablation_model.NeuroFetalNet_ca import NeuroFetalNet_ca
from ablation_model.NeuroFetalNet_without_fusion_3 import NeuroFetalNet_without_fusion_3
from ablation_model.NeuroFetalNet_without_fusion_9 import NeuroFetalNet_without_fusion_9
from ablation_model.NeuroFetalNet_without_pe import NeuroFetalNet_without_pe
from ablation_model.ResNet import ResNet
from ablation_model.ResNet_BiGRU import ResNet_BiGRU
from ablation_model.CNN import CNN
from benchmark_model.Informer import Informer
from benchmark_model.Nonstationary_Transformer import Nonstationary_Transformer
from benchmark_model.TimesNet import TimesNet
from ablation_model.CNN_BiGRU import CRNN
from ablation_model.NeuroFetalNet_with_BiGRU import NeuroFetalNet_BiGRU


def select_model(args, device):
    if args.model == "BiGRU":
        model = BiGRU(num_classes=args.num_classes, input_size=args.in_channels)
    elif args.model == "BiLSTM":
        model = BiLSTM(num_classes=args.num_classes, input_size=args.in_channels)
    elif args.model == "NeuroFetalNet":
        model = NeuroFetalNet(
            num_classes=args.num_classes, in_channels=args.in_channels
        )
    elif args.model == "NeuroFetalNet_without_ca":
        model = NeuroFetalNet_ca(
            num_classes=args.num_classes, in_channels=args.in_channels
        )
    elif args.model == "NeuroFetalNet_without_fusion_3":
        model = NeuroFetalNet_without_fusion_3(
            num_classes=args.num_classes, in_channels=args.in_channels
        )
    elif args.model == "NeuroFetalNet_without_fusion_9":
        model = NeuroFetalNet_without_fusion_9(
            num_classes=args.num_classes, in_channels=args.in_channels
        )
    elif args.model == "NeuroFetalNet_without_pe":
        model = NeuroFetalNet_without_pe(
            num_classes=args.num_classes, in_channels=args.in_channels
        )
    elif args.model == "ResNet":
        model = ResNet(
            num_classes=args.num_classes,
            in_channels=args.in_channels,
            kernel_size=args.kernel_size,
        )
    elif args.model == "ResNet_BiGRU":
        model = ResNet_BiGRU(num_classes=args.num_classes, in_channels=args.in_channels)
    elif args.model == "CNN":
        model = CNN(num_classes=args.num_classes, in_channels=args.in_channels)
    elif args.model == "Informer":
        model = Informer(args)
    elif args.model == "Nonstationary_Transformer":
        model = Nonstationary_Transformer(args)
    elif args.model == "TimesNet":
        model = TimesNet(args)
    elif args.model == "CRNN":
        model = CRNN(num_classes=args.num_classes, in_channels=args.in_channels)
    elif args.model == "NeuroFetalNet_BiGRU":
        model = NeuroFetalNet_BiGRU(
            num_classes=args.num_classes, in_channels=args.in_channels
        )

    model = model.to(device)

    return model
