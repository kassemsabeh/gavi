DATA_DIRECTORY="./datasets"
MODEL_SAVE_DIRECTORY="./saved_models"

python3 run.py \
--file_path="${DATA_DIRECTORY}/az_base_dataset.jsonl" \
--output_model_path="${MODEL_SAVE_DIRECTORY}"