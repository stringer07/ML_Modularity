from datasets import load_from_disk, concatenate_datasets, DatasetDict
from transformers import (T5Tokenizer, T5ForConditionalGeneration, T5Config,
                          TrainingArguments, DataCollatorForSeq2Seq, Trainer)
from pathlib import Path
import torch
from func import CustomTrainer  
import os

use_data_cache = True
use_ckpoint = False
model_name = "flan-t5-base-new"
model_path = Path('./models')
data_path = Path('./data/train/random_ns_nt/preprocessed_data')

lr = 1e-4

epoch = 32

batchsize = 12 
gradient_accumulation_steps = 1
max_token = 10240


save_after_steps = 5000
ckpoint_limit = 2
num_workers = 16

MAX_ALLOWED_LENGTH = 1024


torch.multiprocessing.set_sharing_strategy('file_system')


if use_ckpoint:
    ckpoint_path = './ckpoints/checkpoint-120000'
    print(f"Loading model from :{ckpoint_path}")
    model = T5ForConditionalGeneration.from_pretrained(ckpoint_path)
    tokenizer = T5Tokenizer.from_pretrained('./models/flan-t5-base-new')
    print('max length: {}'.format(model.config.n_positions))

else:
    print("Extending model to support 1024 tokens...")
    
    model = T5ForConditionalGeneration.from_pretrained('./models/flan-t5-base-new')
    model.config.n_positions = 1024
    tokenizer = T5Tokenizer.from_pretrained('./models/flan-t5-base-new')
    print(tokenizer.model_max_length)
    tokenizer.model_max_length = 1024
    print('after extending, max length: {}'.format(tokenizer.model_max_length))

    if len(tokenizer) != model.config.vocab_size:
        print("Resizing model embeddings to match tokenizer size from {} to {}...".format(model.config.vocab_size, len(tokenizer)))
        model.resize_token_embeddings(len(tokenizer))

    print("Fixing zero embeddings for custom tokens...")
    key_tokens = ["egamma", "pow", "INT+", "INT-", "add", "mul", "z", "t", "s"]
    token_ids = tokenizer.convert_tokens_to_ids(key_tokens)
    
    embed_std = model.shared.weight.std().item()

    with torch.no_grad():
        for token_id in token_ids:
            if model.shared.weight[token_id].abs().sum() == 0:
                model.shared.weight[token_id] = torch.randn(model.config.d_model) * (embed_std * 0.5)
                if hasattr(model, 'lm_head') and model.lm_head.weight is not model.shared.weight:
                    model.lm_head.weight[token_id] = model.shared.weight[token_id]
    
    print(f"  Fixed {len(token_ids)} token embeddings: {key_tokens}")


if use_data_cache:
    dataset = load_from_disk(str(data_path))
    print("Data1 cache found in {}!".format(data_path))
    tokenized_dataset = DatasetDict({
        "train": dataset["train"],
        "eval": dataset["eval"]
    })

    print(f"Filtering data longer than {MAX_ALLOWED_LENGTH} tokens...")
    
    def filter_long_data(example):
        return len(example['input_ids']) <= MAX_ALLOWED_LENGTH

    original_train_len = len(tokenized_dataset['train'])
    tokenized_dataset = tokenized_dataset.filter(filter_long_data, num_proc=num_workers)
    
    filtered_train_len = len(tokenized_dataset['train'])
    dropped_count = original_train_len - filtered_train_len
    
    print(f"Filter Complete! Dropped {dropped_count} examples.")
    print(f"Train set size: {original_train_len} -> {filtered_train_len}")

    
else:
    print("Data cache not found!")


data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding='longest',
    pad_to_multiple_of=64
)


training_args = TrainingArguments(
    output_dir="./ckpoints",
    eval_strategy="steps",
    eval_steps=save_after_steps,
    learning_rate=lr,
    
    num_train_epochs=epoch, 
    per_device_train_batch_size=batchsize, 
    per_device_eval_batch_size=batchsize * 2, 
    gradient_accumulation_steps=gradient_accumulation_steps,
    
    max_grad_norm=1,  
    warmup_ratio=0.1,   
    weight_decay=0.01,
    logging_dir="~/tf-logs",
    
    bf16=True,                    
    torch_compile=True,  
    # group_by_length=True,
    # gradient_checkpointing=True,
    
    # optim="adafactor",           

    dataloader_num_workers=num_workers,
    dataloader_pin_memory=True,
    dataloader_persistent_workers=True, 
    dataloader_prefetch_factor=4,
    
    save_steps=save_after_steps,
    save_total_limit=ckpoint_limit,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False
)

trainer = CustomTrainer(
    max_tokens_per_batch= max_token,  
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["eval"],
    data_collator=data_collator,

)


if use_ckpoint:
    print('Start training with checkpoint from : {}'.format(ckpoint_path))
    trainer.train(resume_from_checkpoint=ckpoint_path)
else:
    print('Start training without checkpoint')
    trainer.train()


output_dir = './results'
os.makedirs(output_dir, exist_ok=True)
print("Saving final model and training states...")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)