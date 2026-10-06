# Hugging Face project - Technova AI assistant
#
# Demonstrates: datasets, tokenizers, text generation, the Trainer API,
# PEFT / LoRA, and extractive question answering (all CPU friendly).

import os
import gc
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForQuestionAnswering,
    pipeline,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

from peft import LoraConfig, TaskType, get_peft_model

# setup
print("PyTorch: ", torch.__version__)
print("CPU threads: ", torch.get_num_threads())

# technova data
texts = [
    "Technova is based in Hyderabad",
    "Technova was founded in 2018",
    "Technova develops artificial intelligence software",
    "Technova develops cybersecurity software",
    "Technova provides Python training",
    "Technova provides machine learning training"
]

dataset = Dataset.from_dict({"text": texts})

print("\nDataset: ")
print(dataset)
print(dataset[0])

# tokenizer

model_name = "sshleifer/tiny-gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

text = "Technova develops artificial intelligence"

tokens = tokenizer.tokenize(text)
token_ids = tokenizer.convert_tokens_to_ids(tokens)

print("\nTokens: ", tokens)
print("Token Ids: ", token_ids)

# tokenize dataset
def tokenize_data(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=16
    )

tokenized_dataset = dataset.map(
    tokenize_data,
    batched=True
)

print("\nTokenized Dataset: ")
print(tokenized_dataset[0])

# transformer model
model = AutoModelForCausalLM.from_pretrained(model_name)
model.config.pad_token_id = tokenizer.pad_token_id
print("\nTransformer model loaded")

# text generation pipeline

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=-1
)

result = generator(
    "Technova is",
    max_new_tokens=10,
    do_sample=False
)

print("\n Generated text:")
print(result[0]["generated_text"])

del generator
gc.collect()

# trainer API

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

training_args = TrainingArguments(
    output_dir="./technova_model",
    per_device_train_batch_size=1,
    num_train_epochs=1,
    save_strategy="no",
    logging_steps=1,
    report_to="none",
    use_cpu=True,
    dataloader_num_workers=0
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

print("\nTrainer created")

# PEFT

print("\nPEFT: ")
print("Parameter Efficient Fine-Tuning")
print("Trains only a small part of a pretrained model")

# LoRA

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=4,
    lora_alpha=8,
    lora_dropout=0.05,
    target_modules=["c_attn"]
)

lora_model = get_peft_model(
    model,
    lora_config
)

print("\nLoRA model:")
lora_model.print_trainable_parameters()

# LoRA + Trainer

lora_trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

print("\nLoRA trainer created")

# training intentionally not executed
# lora_trainer.train()

# QLoRA (not executed)
#
#   QLoRA = Quantized model + LoRA adapter
#
# Typical configuration:
#
#   from transformers import BitsAndBytesConfig
#
#   BitsAndBytesConfig(
#       load_in_4bit=True,
#       bnb_4bit_quant_type="nf4"
#   )
#
# QLoRA is not executed here because it is designed for GPU-only hardware.

# release GPT model

del trainer
del lora_trainer
del lora_model
del model
del tokenizer
del data_collator

gc.collect()

print("\nGPT model released")


# question answering

qa_model_name = "distilbert-base-cased-distilled-squad"
qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)
qa_model.eval()

context = """
Technova is a software company based in Hyderabad.
Technova was founded in 2018
Technova develops artificial intelligence software
Technova provides Python and machine learning training
"""

question = "When was Technova founded?"

inputs = qa_tokenizer(
    question,
    context,
    return_tensors="pt"
)

with torch.no_grad():
    outputs = qa_model(**inputs)

start = torch.argmax(outputs.start_logits)
end = torch.argmax(outputs.end_logits)

answer_tokens = inputs["input_ids"][0, start:end + 1]

answer = qa_tokenizer.decode(
    answer_tokens,
    skip_special_tokens=True
)

print("\nQuestion:", question)
print("Answer: ", answer)

# question answering pipeline
print("\nQuestion answering pipeline: ")

# pipeline(
#     "question-answering",
#     model=qa_model_name
# )
