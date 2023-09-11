import json

import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

# Function to split dataset
def split_random(path: str):
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    test, validation = train_test_split(test, test_size=0.5, random_state=42)
    return train, validation, test

# Function to load the dataset in DatasetDict format for train, val, and test
def load_dataset(path: str) -> DatasetDict:
    train, validation, test = split_random(path)
    dataset = DatasetDict()
    dataset['train'] = Dataset.from_pandas(pd.DataFrame(data=train))
    dataset['validation'] = Dataset.from_pandas(pd.DataFrame(data=validation))
    dataset['test'] = Dataset.from_pandas(pd.DataFrame(data=test))
    return dataset

