# BiGRU.py   NeuroFetalNet.py                   NeuroFetalNet_without_pe.py
# BiLSTM.py  NeuroFetalNet_without_ca.py        ResNet_BiGRU.py
# GRU.py     NeuroFetalNet_without_fusion_3.py  ResNet.py
# LSTM.py    NeuroFetalNet_without_fusion_9.py

#     # basic config
    # parser.add_argument(
    #     "--model", type=str, required=True, default="MyNet_4", help="model name"
    # )

    # # gpu
    # parser.add_argument("--use_gpu", type=bool, default=True, help="use gpu or not")
    # parser.add_argument("--gpu", type=str, default="0", help="gpu id")

    # # train
    # parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    # parser.add_argument("--num_epochs", type=int, default=100, help="number of epochs")
    # parser.add_argument("--patience", type=int, default=10, help="patience")

    # # model
    # parser.add_argument("--num_classes", type=int, default=1, help="number of classes")
    # parser.add_argument("--in_channels", type=int, default=1, help="number of channels")
    # parser.add_argument("--seq_len", type=int, default=4800, help="sequence length")

    # args = parser.parse_args()

export CUDA_VISIBLE_DEVICES=0

python -u main.py \
    --model CRNN \
    --input_feature ucp \
    --in_channels 1 \
    --kernel_size 3

python -u main.py \
    --model ResNet \
    --input_feature ucp \
    --in_channels 1 \
    --kernel_size 3

python -u main.py \
    --model ResNet \
    --input_feature ucp \
    --in_channels 1 \
    --kernel_size 5

python -u main.py \
    --model ResNet \
    --input_feature ucp \
    --in_channels 1 \
    --kernel_size 7

python -u main.py \
    --model ResNet \
    --input_feature ucp \
    --in_channels 1 \
    --kernel_size 9

python -u main.py \
    --model CNN \
    --input_feature ucp \
    --in_channels 1

python -u main.py \
    --model ResNet_BiGRU \
    --input_feature ucp \
    --in_channels 1

# python -u main.py \
#     --model BiGRU \
#     --input_feature ucp \
#     --in_channels 1

python -u main.py \
    --model NeuroFetalNet_without_ca \
    --input_feature ucp \
    --in_channels 1

python -u main.py \
    --model NeuroFetalNet \
    --input_feature ucp \
    --in_channels 1