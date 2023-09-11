from tqdm import tqdm
from sklearn import metrics
import torch
import pandas as pd

# pad texts to the same length
def preprocess_test(example, tokenizer, config):
  model_inputs = tokenizer(example['description_category'], max_length=config['max_input_length'], truncation=True,
                           padding="max_length")
  return model_inputs

def test_model(model, dataset, config, tokenizer):
  model.eval()
  for cat in config['categories']:
    # get test split
    test_tokenized_dataset = dataset[cat]

    test_tokenized_dataset = test_tokenized_dataset.map(preprocess_test, batched=True, fn_kwargs={"tokenizer": tokenizer, "config":config})

    # prepare dataloader
    test_tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    dataloader = torch.utils.data.DataLoader(test_tokenized_dataset, batch_size=config['batch_size'])

    with torch.no_grad():
      # generate text for each batch
      all_predictions = []
      for i,batch in enumerate(tqdm(dataloader)):
        batch = {k:v.to(config['device']) for k,v in batch.items()}
        predictions = model.generate(**batch, num_beams=8, do_sample=True, max_length=10)
        decoded_output = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        all_predictions.append(decoded_output)
    all_predictions_flattened = [pred for preds in all_predictions for pred in preds]

    # log predictions as csv file
    zipped_predictions = list(zip(dataset[cat]['description_category'], dataset[cat]['brand'], all_predictions_flattened))
    df = pd.DataFrame(zipped_predictions, columns=['description', 'truth', 'prediction'])
    df.to_csv(f"results.csv")
    # calculate and log accuracy of predictions
    accuracy = metrics.accuracy_score(dataset[cat]['brand'], all_predictions_flattened)
    print(f"Accuracy: {accuracy}")
    print('*' * 200)
    return accuracy