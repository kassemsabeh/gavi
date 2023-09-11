# GAVI: A Category-Aware Generative Approach for Brand Value Identification

This repository contains the source code used in our paper: GAVI: A Category-Aware Generative Approach for Brand Value Identification.

## Installation
1. Clone the repository
2. Download the [openbrand-dataset](https://github.com/kassemsabeh/open-brand/) and place the `az_base_dataset.jsonl` file in the ```datasets``` folder of this repo.
3. Install the required dependencies in the ```requirements.txt``` file:
    ```
    $ pip install -r requirements.txt
    ```


## Model Training
To train the model for the brand value identification task using the ```./datasets/az_base_dataset.jsonl```, run the following shell script:

```
$ bash ./train.sh
```
The model uses the pre-trained ```t5-base``` model from the in the 🤗 Transformers by default to train the model. The trained model will be stored in ```./saved_models/```.

## Download Pre-trained Model
We provide a pre-trained model on the `az_base_dataset.jsonl` dataset [here](https://www.dropbox.com/sh/v7fnczj2ykfxuwl/AAC9gRSZrH5TbIRDU2x7CZnTa?dl=0). Download the folder and place it in the ```saved_models``` folder of the repo.

After running all scripts, you should obtain the following directory tree:
```bash
├── README.md
├── config.py
├── datasets
│   └── az_base_dataset.jsonl
├── saved_models
│   └── gavi
│       └── config.json
│       └── generation_config.json
│       └── pytorch_model.bin
├── data.py
├── train.py
├── requirements.txt
├── train.sh
├── test.py
```