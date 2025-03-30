import json
import torch
from evaluate import load
from transformers import AutoTokenizer, AutoModelForCausalLM

def evaluate_model(model_path: str, test_data_path: str) -> float:
    # Load test data
    with open(test_data_path, "r") as f:
        test_dict = json.load(f)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load accuracy metric
    accuracy_metric = load("accuracy")
    predictions, references = [], []

    # Evaluate each prompt
    for prompt, expected in test_dict.items():
        # Tokenize input and move to GPU
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Generate output
        output_ids = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id
        )
        
        # Decode and process generated text
        generated = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        if generated.startswith(prompt):
            generated = generated[len(prompt):].strip()

        predictions.append(generated)
        references.append(expected.strip())

    # Compute accuracy
    accuracy = accuracy_metric.compute(predictions=predictions, references=references)["accuracy"]
    return round(accuracy * 100, 1)