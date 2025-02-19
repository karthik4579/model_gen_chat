import os
import uuid
import json
import torch
from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth.chat_templates import get_chat_template, train_on_responses_only

def finetune_model(model_name, dataset_path,dataset_type):
    if dataset_type == "text_only":
        supported_models = {
            "Qwen2.5-3B": "unsloth/Qwen2.5-3B-Instruct",
            "Llama-3.2B": "unsloth/Llama-3.2-3B-Instruct",
            "Qwen2.5-7B": "unsloth/Qwen2.5-7B-Instruct",
        }
        
        if model_name not in supported_models:
            raise ValueError(f"Model {model_name} not supported. Choose from {list(supported_models.keys())}")

        # Load and format dataset
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        # Convert to conversational format
        formatted_data = [{
            "conversations": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response}
            ]
        } for prompt, response in data.items()]

        # Load model and tokenizer with correct parameters (passing the checkpoint path as a positional argument)
        model, tokenizer = FastLanguageModel.from_pretrained(
            supported_models[model_name],
            max_seq_length=2048,
            load_in_4bit=True,
            device_map="auto",
        )

        # Apply appropriate chat template
        tokenizer = get_chat_template(
            tokenizer,
            chat_template="llama-3.1" if "Llama" in model_name else "qwen2.5"
        )

        # Create HuggingFace dataset
        dataset = Dataset.from_list(formatted_data)
        
        # Format prompts using the tokenizer's chat template
        def formatting_prompts_func(examples):
            texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) 
                    for convo in examples["conversations"]]
            return {"text": texts}
        
        dataset = dataset.map(formatting_prompts_func, batched=True)

        # Configure LoRA adapters (this applies parameter-efficient fine-tuning)
        model = FastLanguageModel.get_peft_model(
            model,
            r=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=32,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
        )

        # Updated training configuration based on recommendations:
        training_args = TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=15,            # Increased warmup steps for stability
            max_steps=50,              # Extended training steps to allow more adaptation
            learning_rate=1e-5,         # Lower learning rate for better convergence with synthetic data
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=25,
            output_dir="outputs",
            optim="adamw_8bit",
            save_strategy="steps",
            save_steps=50,              # Adjusted save frequency
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            report_to="none",
        )

        # Create trainer with a proper data collator
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=2048,
            data_collator=DataCollatorForSeq2Seq(
                tokenizer=tokenizer,
                pad_to_multiple_of=8,
                return_tensors="pt",
                padding=True
            ),
            packing=False,
        )

        # Train only on assistant responses
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
            response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
        )

        # Execute training
        trainer.train()

        # Save the model directly in GGUF format with full precision and 4-bit quantization
        os.makedirs("finetuned_models", exist_ok=True)
        model_filename = f"{model_name}_{uuid.uuid4()}"
        model_path = os.path.join("finetuned_models", model_filename)
        
        model.save_pretrained_gguf(
            model_path,
            tokenizer,
            quantization_method="q4_k_m",
            maximum_memory_usage=0.7  # Adjust based on your available memory
        )

        return model_path

