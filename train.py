from config import config
from test import test_model
from data import load_dataset
from absl import flags

import os
import sys
import time

from tqdm import tqdm
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration, DataCollatorForSeq2Seq,Seq2SeqTrainingArguments, Seq2SeqTrainer
from torch.utils.data import DataLoader

_OUTPUT_MODEL_NAME = flags.DEFINE_string(
    'output_model_path',
    default=None,
    help='The output name for saving the model',
    required=True
)

_INPUT_DATA_DIR  = flags.DEFINE_string(
    'data_directory',
    default=None,
    help='The input directory for the datasets',
    required=True
)

_INPUT_FILE_PATH = flags.DEFINE_string(
    'file_path',
    default=None,
    help='The input directory for the train jsonl file',
    required=True
)

FLAGS = flags.FLAGS
FLAGS(sys.argv)

config['dataset_path'] = _INPUT_FILE_PATH.value
config['save_path'] = _OUTPUT_MODEL_NAME.value

# load datasets
dataset = load_dataset(config['dataset_path'])

tokenizer = AutoTokenizer.from_pretrained(config['model_id'])
tokenizer.add_special_tokens({'additional_special_tokens': list(config['ADDITIONAL_SP_TOKENS'].values())})

# define the preprocessing function
def preprocess_function(example):
  # highlight category inside input
  example['description_category'] = example['description'] + f" {config['ADDITIONAL_SP_TOKENS']['hl']} {example['category']} {config['ADDITIONAL_SP_TOKENS']['hl']}"
  model_inputs = tokenizer(example['description_category'], max_length=config['max_input_length'], truncation=True)
  with tokenizer.as_target_tokenizer():
    labels = tokenizer(example['brand'], max_length=config['max_target_length'], truncation=True)
  
  model_inputs['labels'] = labels['input_ids']
  return model_inputs

# preprocess and tokenize the dataset
processed_datasets = dataset.map(preprocess_function, batched=True).remove_columns(dataset["train"].column_names)

# initilaize model
model = T5ForConditionalGeneration.from_pretrained(config['model_id'])
model.resize_token_embeddings(len(tokenizer))

model = model.to(config['device'])
# define the optimizer
optimizer = torch.optim.Adam(params =  model.parameters(), lr=config['learning_rate'])

# define data collator
data_collator = DataCollatorForSeq2Seq(tokenizer)

train_dataloader = DataLoader(processed_datasets['train'], shuffle=True, collate_fn=data_collator, batch_size=config['batch_size'])
validation_dataloader = DataLoader(processed_datasets['validation'], shuffle=False, collate_fn=data_collator, batch_size=config['batch_size'])

def train(n_epochs, tokenizer, model, train_loader, validation_loader, optimizer):
  min_loss = 10e3
  best_epoch = 0
  tolerance = 4
  for epoch in range(n_epochs):
    epoch_start_time = time.time()
    model.train()
    total_loss = 0
    for step, batch in enumerate(tqdm(train_loader)):
        batch = {k: v.to(config['device']) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.detach().float()
    new_loss = evaluate_model(model, tokenizer, validation_loader)
    train_epoch_loss = total_loss / len(train_loader)
    if new_loss < min_loss:
            min_loss = new_loss
            best_epoch = epoch
            save_model(model)
    else:
       tolerance -= 1
    elapsed_time = time.time() - epoch_start_time
    print('-' * 200)
    print(f'| Epoch {epoch:3d} | Time: {elapsed_time:5.2f}s | Train Loss: {train_epoch_loss:.4f} |  Validation Loss: {new_loss:.4f} | Best Epoch: {best_epoch}')
    print('-' * 200)

    if tolerance == 0:
        print('*' * 200)
        print(f"Early stopping applied after {epoch} epochs. Best Epoch: {best_epoch} Best Loss: {min_loss}")
        return model
  return model

def evaluate_model(model, tokenizer, val_loader):
    model.eval()
    eval_loss = 0
    eval_preds = []
    for step, batch in enumerate(tqdm(val_loader)):
        batch = {k: v.to(config['device']) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss
        eval_loss += loss.detach().float()
        # eval_preds.extend(
        #     tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True)
        # )
    eval_epoch_loss = eval_loss / len(val_loader)
    return eval_epoch_loss

def save_model(model):
    model.save_pretrained(config['save_path'], from_pt=True)


def main():
    if not os.path.exists(_OUTPUT_MODEL_NAME.value):
        os.mkdir(_OUTPUT_MODEL_NAME.value)
    
    if not os.path.exists(_INPUT_DATA_DIR.value):
        os.mkdir(_INPUT_DATA_DIR.value)
    
    # Build, train and analyze the model with the pipeline
    model = train(config['n_epochs'], tokenizer, model, train_dataloader, validation_dataloader, optimizer)
    test_model(model, config, tokenizer)

if __name__ == '__main__':
    main()