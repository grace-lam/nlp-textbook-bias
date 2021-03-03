import os
import math
import torch

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForMaskedLM
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

from finetune_bert import *

test_file = 'bert_data/80_10_10/test_textbook_data.txt'
pretrained_checkpoint = 'bert-base-uncased'
finetuned_checkpoint = 'bert_mlm/80_10_10/bert_mlm_textbook'
block_size = 128

def tokenize_function(examples):
    return tokenizer(examples["text"])

datasets = load_dataset('text', data_files={"test": test_file})
tokenizer = AutoTokenizer.from_pretrained(pretrained_checkpoint)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    batch_size=1000,
    num_proc=4,
)

def evaluate_bert(model_checkpoint):
    model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
    trainer = Trainer(
        model=model,
        eval_dataset=lm_datasets["test"],
        data_collator=data_collator,
    )
    loss = trainer.evaluate()
    return loss

pretrained_test_loss = evaluate_bert(pretrained_checkpoint)
print('Pretrained BERT:')
print(f"Cross-entropy Loss: {pretrained_test_loss['eval_loss']:.2f}")
print(f"Perplexity: {math.exp(pretrained_test_loss['eval_loss']):.2f}")

finetuned_test_loss = evaluate_bert(finetuned_checkpoint)
print('Finetuned BERT:')
print(f"Cross-entropy Loss: {finetuned_test_loss['eval_loss']:.2f}")
print(f"Perplexity: {math.exp(finetuned_test_loss['eval_loss']):.2f}")

