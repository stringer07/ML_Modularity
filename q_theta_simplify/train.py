from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from pathlib import Path
import torch
import os


use_data_cache = True
model_path = Path('./models')

pretrained_mdl = "t5-small" 

lr = 1e-4  
batchsize = 128  
epoch = 64
save_after_steps = 5000
ckpoint_limit = 2 
num_workers = 16 



tokenizer = T5Tokenizer.from_pretrained(model_path / "t5-small-new")
print(f"loading {pretrained_mdl} model...")
if pretrained_mdl == "t5-small":
    model = T5ForConditionalGeneration.from_pretrained(model_path / "t5-small-new")

elif pretrained_mdl == "t5-base":
    model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-base")


print(f"Resizing model embeddings to match tokenizer {model.shared.weight.shape}...")
model.resize_token_embeddings(len(tokenizer))
print(f"Model embeddings resized to: {model.shared.weight.shape}") 


print("Fixing zero embeddings for custom tokens...")
key_tokens = ["q_theta", "pow", "INT+", "INT-", "add", "mul", "z", "t"]
token_ids = tokenizer.convert_tokens_to_ids(key_tokens)
embed_std = model.shared.weight.std().item()

with torch.no_grad():
    for token_id in token_ids:
        model.shared.weight[token_id] = torch.randn(model.config.d_model) * (embed_std * 0.02)
        if hasattr(model, 'lm_head') and model.lm_head.weight is not model.shared.weight:
            model.lm_head.weight[token_id] = model.shared.weight[token_id]

print(f"  Fixed {len(token_ids)} token embeddings: {key_tokens}")

def fix_labels_padding(example):
    example['labels'] = [
        token if token != tokenizer.pad_token_id else -100 
        for token in example['labels']
    ]
    return example
    
if use_data_cache:
    print("loading preprocessed data ...")
    tokenized_dataset = load_from_disk("./data/train/random_ns_nt/preprocessed_data")
    
    print("fixing labels padding (0 -> -100)...")
    tokenized_dataset = tokenized_dataset.map(fix_labels_padding, num_proc=num_workers)

print('collating data...')
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer, 
    model=model, 
    padding='longest', 
    # max_length=512,
    pad_to_multiple_of=8
)

training_args = TrainingArguments(
    output_dir="./ckpoints",
    eval_strategy="steps",
    eval_steps=save_after_steps,
    learning_rate=lr,
    
    num_train_epochs=epoch, 
    per_device_train_batch_size=batchsize, 
    per_device_eval_batch_size=batchsize * 2, 

    max_grad_norm=1,  
    warmup_ratio=0.1,   
    weight_decay=0.01,
    logging_dir="/tf-logs",
    
    bf16=True,                    
    torch_compile=True,  
    group_by_length=True,
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

trainer = Trainer(
    model=model, 
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["eval"],
    data_collator=data_collator,
)

print("Starting training on RTX 5090 Beast Mode...")
trainer.train()

model.save_pretrained("./results")
tokenizer.save_pretrained("./results")