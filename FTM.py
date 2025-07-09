import os
import transformers
import torch
from datasets import load_dataset
from trl import SFTTrainer
from peft import LoraConfig
from transformers import TrainingArguments
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig, GemmaTokenizer

hf_token = "YOUR_TOKEN_KEY"

model_id = "google/gemma-7b"

from transformers import BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=hf_token)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    offload_folder="offload",
    use_auth_token=hf_token,
    low_cpu_mem_usage=True,
)

dataset = load_dataset("json", data_files={"train": "YOUR_DATASET.json"})

def tokenize_function(examples):
    
    texts = []
    for i in range(len(examples["input"])):
        combined = f"Input: {examples['input'][i]} Output: {examples['output'][i]}"
        texts.append(combined)
    return tokenizer(texts, truncation=True, max_length=128, padding="max_length")

tokenized_datasets = dataset.map(tokenize_function, batched=True)

lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=2,  
    gradient_accumulation_steps=4,  
    learning_rate=2e-4,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=500,
    report_to="none",

    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant': False}
)

trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_datasets["train"],
    peft_config=lora_config,
    args=training_args,
)

trainer.train()

output_dir = "./finetuned_model"
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

base_model_id = "google/gemma-7b"

del model
import gc
gc.collect()
torch.cuda.empty_cache()

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=bnb_config,
    offload_folder="offload",  
)

peft_model_id = "./finetuned_model"

from peft import PeftModel
from transformers import AutoTokenizer 

model = PeftModel.from_pretrained(base_model, peft_model_id)

tokenizer = AutoTokenizer.from_pretrained(peft_model_id)

model = model.merge_and_unload()

def generate_text(query, model, tokenizer, max_length=200):
    inputs = tokenizer(query, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=max_length, num_return_sequences=1)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

def interactive_chat(model, tokenizer):
    print("Enter 'exit' to quit the chat.")
    while True:
        query = input("You: ")
        if query.lower() == 'exit':
            print("Exiting chat.")
            break
        response = generate_text(query, model, tokenizer)
        print(f"Model: {response}\n")

if __name__ == "__main__":
    interactive_chat(model, tokenizer)
