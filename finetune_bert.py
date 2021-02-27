import os
import torch

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForMaskedLM
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

train_file = 'train_textbook_data.txt'
eval_file = 'eval_textbook_data.txt'
model_checkpoint = 'bert-base-uncased'
model_dir='bert_mlm/'
block_size = 128

def gpu_check():
    # If there's a GPU available...
    if torch.cuda.is_available():    
        # Tell PyTorch to use the GPU.    
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    # If not...
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

def load_data():
    def tokenize_function(examples):
        return tokenizer(examples["text"])

    datasets = load_dataset('text', data_files={"train": train_file, "validation": eval_file})
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        batch_size=1000,
        num_proc=4,
    )
    return lm_datasets, data_collator

def finetune_bert(lm_datasets, data_collator):
    model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
    
    training_args = TrainingArguments(
        output_dir=model_dir+'train',
        logging_dir=model_dir+'logs',
        evaluation_strategy='steps',
        eval_steps=4000,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["validation"],
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(model_dir+'bert_mlm_textbook')

def main():
    gpu_check()
    lm_datasets, data_collator = load_data()
    os.makedirs(model_dir, exist_ok=True)
    finetune_bert(lm_datasets, data_collator)

if __name__ == '__main__':
    main()
