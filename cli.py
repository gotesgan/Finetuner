import argparse
import os
from finetune import finetune_model

def main():
    parser = argparse.ArgumentParser(description='Fine-tune a language model on a custom dataset.')
    parser.add_argument('model_name', type=str, help='Name or path of the pre-trained model')
    parser.add_argument('dataset_path_or_name', type=str, help='Name or path of the dataset')
    parser.add_argument('-m', '--max_seq_length', type=int, default=1024, help='Maximum sequence length for the model (default: 1024)')
    parser.add_argument('-o', '--output_dir', type=str, default=None, help='Directory to save the fine-tuned model (default: ./fine-tuned-<model_name>)')
    parser.add_argument('-t', '--tokenizer_name', type=str, default=None, help='Name or path of the tokenizer (optional)')
    parser.add_argument('-e', '--num_epochs', type=int, default=3, help='Number of epochs for fine-tuning (default: 3)')
    parser.add_argument('-b', '--batch_size', type=int, default=8, help='Batch size for fine-tuning (default: 8)')
    parser.add_argument('-l', '--learning_rate', type=float, default=5e-5, help='Learning rate for fine-tuning (default: 5e-5)')
    parser.add_argument('-f', '--finetune_technique', type=str, choices=['standard', 'sft', 'dpo'], default='standard', help='Fine-tuning technique to use (default: standard)')

    args = parser.parse_args()

    if args.output_dir is None:
        model_name = os.path.basename(args.model_name)
        args.output_dir = f'./fine-tuned-{model_name}'

    finetune_model(
        model_name=args.model_name,
        dataset_path_or_name=args.dataset_path_or_name,
        max_seq_length=args.max_seq_length,
        output_dir=args.output_dir,
        tokenizer_name=args.tokenizer_name,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        finetune_technique=args.finetune_technique,
    )

if __name__ == '__main__':
    main()
