import requests
import json
import re
from pathlib import Path
from langchain_core.prompts import PromptTemplate
import asyncio

with open(f"{Path.cwd()}/prompts/prompt_gen.txt") as prompt:
    raw_system_prompt = prompt.read()

system_prompt = PromptTemplate.from_template(raw_system_prompt)

def generate_prompts_list(dataset_goal,seed_prompts_val="",should_classify_val="false"):
    final_system_prompt = system_prompt.format(seed_prompts=seed_prompts_val,should_classify=should_classify_val)
    
    payload = {
        "messages": [
            {"role": "system", "content": f"{final_system_prompt}"},
            {"role": "user", "content": dataset_goal}
        ],
        "max_tokens": 28000,
        "model": "meta-llama/Meta-Llama-3.1-405B-Instruct",
        "temperature": 0.5,
    }

    response = requests.post(
        url="https://deepinfra-wrapper.onrender.com/chat/completions",
        json=payload,
        timeout=300
    ).json()
    
    raw_data =  response['choices'][0]['message']['content']
    pattern = r'<start_json>(.*?)<end_json>'
    actual_json = re.search(pattern, raw_data, re.DOTALL)
    return json.loads(actual_json.group(1))

goal = "Create a dataset of new engine parts with their prices and quantities from the last year. Include information about each part's make, model, year, part number, description, and current market price. The dataset will be used to train a language model to predict engine part prices."
json_data = generate_prompts_list(should_classify_val="true",dataset_goal=goal)

print(json_data)