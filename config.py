import torch

# define hyperparameters
config = dict(
    device = 'cuda' if torch.cuda.is_available() else 'cpu',
    model_id = 't5-base',
    n_epochs = 10,
    max_input_length = 512,
    categories = ['test'],
    ADDITIONAL_SP_TOKENS = {'hl': '<hl>'},
    max_target_length = 20,
    batch_size = 32,
    dataset_path = '/datasets/az_base_dataset.jsonl',
    save_path='/saved_models',
    learning_rate = 5e-5,
)
