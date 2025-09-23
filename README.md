###### [Overview](#NeuroFetalNet) | [Setup & Usage](#Setup-&-Usage) | [Weights](#NeuroFetalNet-weights) | [Acknowledgements](#Acknowledgements)

![License](https://img.shields.io/badge/license-MIT-brightgreen)

# NeuroFetalNet

Code for paper: "*NeuroFetalNet: Advancing Remote Electronic Fetal Monitoring with a New Dataset and Comparative Analysis of FHR and UCP Impact*". 

NeuroFetalNet utilizes multi-scale feature extractor to effectively capture features from FHR (Fetal Heart Rate) and UCP (Uterine Contraction Pattern). This approach has demonstrated SOTA performance in predicting the health status of the fetus.

## Setup & Usage

1. Clone the repository:

   ```
   git clone https://github.com/BlackThompson/NeuroFetalNet.git
   ```

2. Create a new environment and install dependencies:

   - Python version should be >= 3.8.
   - The versions of `torch`, `torchvision`, and `torchaudio` should align with your CUDA version.

   ```
   conda create --name fetalbeat python=3.8
   conda activate fetalbeat
   pip install -r requirements.txt
   cd NeuroFetalNet
   ```

3. Download the dataset from [OneDrive]([BabyBeat_dataset](https://1drv.ms/f/c/01109175203b3107/EgcxOyB1kRAggAHsAAAAAAABeMUEwJF-s-_bxe_QRErNoA?e=gR1zBI)), and replace the folder `BabyBeat_dataset` with the downloaded folder.

4. Run the script `ablation.sh` to reproduce the best results. 

   ```
   bash ablation.sh
   ```

## NeuroFetalNet weights

NeuroFetalNet weights can be downloaded from [OneDrive]([NeuroFetalNet_16bs_100epoc_3ks_both.pth](https://1drv.ms/u/c/01109175203b3107/EQcxOyB1kRAggAHxAAAAAAABgFvWG_iYBkJMl5G-WXRjWg?e=yVxXZJ)).

## Acknowledgements

I would like to express my gratitude to my co-authors [Jiaqi Zhao](https://github.com/baobooooo), [Xinrong Miao](https://github.com/stefenMiao]) and [Yanqiao Wu](https://github.com/wyq666999) for their valuable contributions to this reasearch.
