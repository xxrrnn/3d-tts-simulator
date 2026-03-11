"""
This file is largely borrowed from OpenR (https://github.com/openreasoner/openr)
"""

from pathlib import Path
import jsonlines
from torch.utils.data import Dataset


def get_train_test_dataset(env_name, *args, **kwargs):
    env_dir = Path(__file__).parent
    train_ds = None
    if env_name == 'MATH':
        print(f'Loading MATH dataset')
        test_ds = JsonlMathDataset(env_dir / "dataset/test500.jsonl")
    elif env_name == 'AMC23':
        print(f'Loading AMC23 dataset')
        test_ds = JsonlMathDataset(env_dir / "dataset/test_amc.jsonl")
    elif env_name == 'AIME24':
        print(f'Loading AIME24 dataset')
        test_ds = JsonlMathDataset(env_dir / "dataset/test_aime.jsonl")
    return train_ds, test_ds


class JsonlMathDataset(Dataset):

    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path
        self.data = []
        with jsonlines.open(data_path, "r") as reader:
            for obj in reader:
                self.data.append(obj)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        if 'amc' in self.data_path.stem:
            return {"question": x["problem"], "answer": str(x["solution"]), "extracted_groundtruth": str(x["extracted_groundtruth"])}
        elif 'aime' in self.data_path.stem:
            return {"question": x["problem"], "answer": str(x["solution"]), "extracted_groundtruth": str(x["extracted_groundtruth"])}
        else:
            return {"question": x["problem"], "answer": x["solution"], "level": x["level"]}
