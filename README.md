```markdown
# finetuner

`finetuner` is a command-line tool for fine-tuning language models on custom datasets. It supports various fine-tuning techniques, including standard fine-tuning, Supervised Fine-Tuning (SFT), and Dialogue Prompt Optimization (DPO).

## Features

- Fine-tune pre-trained language models from Hugging Face Transformers or large models like Gemma-7B
- Support for standard fine-tuning, SFT, and DPO techniques
- Load datasets from the Hugging Face Hub or local directories
- Customize fine-tuning parameters like sequence length, epochs, batch size, and learning rate
- Save fine-tuned models to a specified output directory

## Installation

You can install the `finetuner` package from PyPI using pip: `pip install finetuner`

## Usage

The `finetuner` command-line tool accepts the following arguments:

```
finetuner model_name dataset_path_or_name [-h] [-m MAX_SEQ_LENGTH] [-o OUTPUT_DIR] [-t TOKENIZER_NAME] [-e NUM_EPOCHS] [-b BATCH_SIZE] [-l LEARNING_RATE] [-f {standard,sft,dpo}]
```

- `model_name`: Name or path of the pre-trained model (required)
- `dataset_path_or_name`: Name or path of the dataset (required)
- `-m`, `--max_seq_length`: Maximum sequence length for the model (default: 1024)
- `-o`, `--output_dir`: Directory to save the fine-tuned model (default: `./fine-tuned-<model_name>`)
- `-t`, `--tokenizer_name`: Name or path of the tokenizer (optional)
- `-e`, `--num_epochs`: Number of epochs for fine-tuning (default: 3)
- `-b`, `--batch_size`: Batch size for fine-tuning (default: 8)
- `-l`, `--learning_rate`: Learning rate for fine-tuning (default: 5e-5)
- `-f`, `--finetune_technique`: Fine-tuning technique to use (default: `standard`, choices: `standard`, `sft`, `dpo`)

### Examples

Standard fine-tuning with default parameters: `finetuner gpt2 dataset/path`

Supervised Fine-Tuning (SFT) with custom parameters: `finetuner unsloth/gemma-7b-bnb-4bit gemma-sft-dataset -f sft -m 2048 -e 5 -b 4 -l 2e-5 -o gemma-sft-model`

Dialogue Prompt Optimization (DPO) with custom tokenizer: `finetuner unsloth/gemma-7b-bnb-4bit gemma-dpo-dataset -f dpo -t custom/tokenizer`

## Creating a pip package from GitHub

If you have the `finetuner` package code in a GitHub repository, you can create a pip package and distribute it on PyPI or other package repositories. Here's how you can do it:

1. Make sure your repository has the following file structure: `finetuner/`, `finetuner/__init__.py`, `finetuner/cli.py`, `finetuner/finetune.py`, `README.md`, `setup.py`, `requirements.txt`
2. Commit and push your changes to the GitHub repository.
3. Install the `build` package: `pip install build`
4. In the root directory of your repository, run `python -m build` to create a source distribution and a binary distribution.
5. Install the `twine` package: `pip install twine`
6. Upload the distribution files to PyPI (or any other package repository) using `twine`: `twine upload dist/*`

After completing these steps, your `finetuner` package will be available on PyPI or the package repository you chose. Users can then install the package using `pip install finetuner`.

Note: Before uploading your package to PyPI, make sure to follow their guidelines and best practices for packaging and distribution.
```

This single file version of the `README.md` file includes all the necessary information for users to understand the `finetuner` package, install it, use it, and create a pip package from the GitHub repository for distribution on PyPI or other package repositories. It covers the package features, installation instructions, usage examples with various fine-tuning techniques, and steps to create and upload the pip package.
