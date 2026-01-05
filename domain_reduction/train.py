from datasets import load_from_disk
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from pathlib import Path
import torch


use_data_cache=True
model_name="t5-small"
model_path=Path('./models')
data_path=Path('./datasets/preprocessed_data/v3')

use_ckpoint=False
batchsize=1024

lr=3e-4
epoch=256
save_after_steps=5000
ckpoint_limit=2


tokenizer = T5Tokenizer.from_pretrained(model_path/model_name)
model = T5ForConditionalGeneration.from_pretrained(model_path/model_name)
print("loading {} model...".format(model_name))   

print(f"Max length: {model.config.n_positions}")
print(f"Embedding shape: {model.get_input_embeddings().weight.shape}")


# if use_ckpoint:
#     tokenizer = T5Tokenizer.from_pretrained("./models/flan-t5-small-new")
#     model = T5ForConditionalGeneration.from_pretrained("./ckpoints/checkpoint-59000")       
#     print("loading ckponit...")

# else :
#     tokenizer = T5Tokenizer.from_pretrained(model_path/model_name)
#     model = T5ForConditionalGeneration.from_pretrained(model_path/model_name)
#     print("loading {} model...".format(model_name))    


if use_data_cache:
    print("loading preprocessed data for {} ...".format(model_name))
    tokenized_dataset = load_from_disk(str(data_path))

else:
    print("Not find data_cache! ")

print('collating data...')
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding='longest', max_length=16)

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.empty_cache()

model = model.to(device)

training_args = TrainingArguments(
    output_dir="./ckpoints",
    eval_strategy="steps",
    eval_steps=5000,
    learning_rate=lr,
    per_device_train_batch_size=batchsize,
    per_device_eval_batch_size=batchsize,
    num_train_epochs=epoch,
    weight_decay=0.01,
    logging_dir="./logs",
    bf16=True,
    save_steps=save_after_steps,
    save_total_limit=ckpoint_limit,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["eval"],
    data_collator=data_collator,
)

trainer.train()
model.save_pretrained("./results")
tokenizer.save_pretrained("./results")