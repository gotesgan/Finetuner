import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset, DatasetDict
from unsloth import FastLanguageModel
from trl import SFTTrainer, DPOTrainer

def finetune_model(model_name, dataset_path_or_name, max_seq_length, output_dir, tokenizer_name=None, num_epochs=3, batch_size=8, learning_rate=5e-5, finetune_technique='standard', **kwargs):
    if tokenizer_name is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    if finetune_technique == 'standard':
        model = AutoModelForCausalLM.from_pretrained(model_name)
    elif finetune_technique in ['sft', 'dpo']:
        model, _ = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=torch.float16 if torch.cuda.is_available() else None,
            load_in_4bit=True,
        )

    if os.path.isdir(dataset_path_or_name):
        # Load dataset from local directory
        data_files = {
            'train': os.path.join(dataset_path_or_name, 'train.txt'),
            'test': os.path.join(dataset_path_or_name, 'test.txt')
        }
        dataset = DatasetDict.load_dataset('text', data_files=data_files)
    else:
        # Load dataset from Hugging Face Hub
        dataset = load_dataset(dataset_path_or_name)

    training_args = TrainingArguments(
        output_dir=output_dir,
        max_seq_length=max_seq_length,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        # Add other training arguments as needed
        **kwargs
    )

    if finetune_technique == 'standard':
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['test'],
            tokenizer=tokenizer,
        )
    elif finetune_technique == 'sft':
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset['train_sft'],
            eval_dataset=dataset['test_sft'],
            max_seq_length=max_seq_length,
            args=training_args
        )
    elif finetune_technique == 'dpo':
        trainer = DPOTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset['train_dpo'],
            eval_dataset=dataset['test_dpo'],
            tokenizer=tokenizer
        )

    trainer.train()
    trainer.save_model(output_dir)
