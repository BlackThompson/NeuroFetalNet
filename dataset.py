from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from sktime.datasets import load_from_tsfile_to_dataframe
import torch


class BabyBeatDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data_X, self.data_y = load_from_tsfile_to_dataframe(data_dir)
        # self.label_encoder = LabelEncoder()
        # self.labels = self.label_encoder.fit_transform(self.data_y)
        self.labels = self._encode_labels(self.data_y)

    def __len__(self):
        return len(self.data_y)

    def __getitem__(self, idx):
        # 转化为tensor,用iloc读取数据

        # fhr进行归一化，mean=142.63593, std=11.838785
        # ucp进行归一化，mean=21.884153, std=17.193254
        fhr = torch.tensor(self.data_X.iloc[idx, 0], dtype=torch.float32)
        fhr = (fhr - 142.63593) / 11.838785
        ucp = torch.tensor(self.data_X.iloc[idx, 1], dtype=torch.float32)
        ucp = (ucp - 21.884153) / 17.193254
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.transform:
            fhr = self.transform(fhr)

        return fhr, ucp, label

    def _encode_labels(self, labels):
        label_mapping = {"1": 1, "0": 0}
        return [label_mapping[label] for label in labels]
