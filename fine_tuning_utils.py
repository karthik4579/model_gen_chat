import os
import uuid
from unsloth import FastLanguageModel
from datasets import Dataset

def finetune_model(model_name):
    supported_models = {
        "Qwen2.5-3B": "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
        "Llama-3.2B": "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
        "Qwen2.5-7B": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    }
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name_or_path=supported_models[model_name],
        max_seq_length=2048,
        load_in_4bit=True,
    )
    
    dataset = [{"text": f"### Prompt:\n{example['prompt']}\n\n### Response:\n{example['response']}"} 
               for example in synthetic_dataset]
    
    trainer = FastLanguageModel.get_trainer(
        model,
        train_dataset=Dataset.from_list(dataset),
        max_seq_length=2048,
        dataset_text_field="text",
    )
    
    trainer.train()
    
    model_path = os.path.join("finetuned_models", f"{model_name}_{uuid.uuid4()}")
    os.makedirs(model_path, exist_ok=True)
    
    model.save_pretrained_gguf(model_path, tokenizer)
    return (model_name, model_path)