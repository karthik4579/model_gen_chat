import os
import requests
import json
import json_repair
from pathlib import Path
from langchain_core.prompts import PromptTemplate
from groq import Groq
from dotenv import dotenv_values
import random
import time
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import uuid
import threading
#from fine_tuning_utils import finetune_model
from sentence_transformers import SentenceTransformer
from torchmetrics.text.bert import BERTScore
from typing import Any

embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

config_values = dotenv_values(f"{Path.cwd()}/config.env")
api_url = config_values['AI_API_URL']
client = Groq(api_key=config_values["AI_API_KEY"])
chat_client = Groq(api_key=config_values["AI_API_KEY_CHAT"])

with open(f"{Path.cwd()}/prompts/prompt_gen.txt") as promptgen_prompt:
    raw_promptgen_system_prompt = promptgen_prompt.read()
with open(f"{Path.cwd()}/prompts/chat_gen.txt") as chatgen_prompt:
    chatgen_system_prompt = chatgen_prompt.read()
with open(f"{Path.cwd()}/prompts/data_gen.txt") as datagen_prompt:
    raw_datagen_system_prompt = datagen_prompt.read()
with open(f"{Path.cwd()}/prompts/rewrite_gen.txt") as rewritegen_prompt:
    raw_rewrite_system_prompt = rewritegen_prompt.read()

promptgen_system_prompt = PromptTemplate.from_template(raw_promptgen_system_prompt)
datagen_system_prompt = PromptTemplate.from_template(raw_datagen_system_prompt)
rewritegen_system_prompt = PromptTemplate.from_template(raw_rewrite_system_prompt)

request_lock = threading.Lock()

def make_sequential_request(endpoint, headers, instance, retry_count=8) -> Any:
    with request_lock:
        for attempt in range(retry_count):
            try:
                response = requests.post(endpoint, headers=headers, json=instance, timeout=3000)
                return response.json()
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < retry_count - 1:
                    continue
                raise

def generate_prompts(dataset_goal, seed_data_val="", dataset_type_val="", should_search_val="true"):
    final_promptgen_system_prompt = promptgen_system_prompt.format(
        seed_data=seed_data_val,
        should_search=should_search_val,
        dataset_type=dataset_type_val
    )

    endpoint = f"{api_url}/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config_values['AI_API_KEY']}"
    }

    instance = {
        "model": "llama-3.3-70b",
        "messages": [
            {"role": "system", "content": final_promptgen_system_prompt},
            {"role": "user", "content": dataset_goal}
        ],
        "stream": False,
        "max_tokens": 8192
    }

    if seed_data_val == "":
        try:
            search_metadata = make_sequential_request(endpoint, headers, instance)
            if isinstance(search_metadata, dict) and 'choices' in search_metadata:
                content = search_metadata['choices'][0]['message']['content']
                print()
                return json_repair.loads(content)
            raise ValueError("Invalid response format")
        except Exception as e:
            print(f"Error in search metadata generation: {e}")
            raise
    else:
        combined_json = {}
        prompt_counter = 1  # Track total prompts across batches

        for i in range(1,5):
            try:
                response = make_sequential_request(endpoint, headers, instance)
                print(response)
                if not isinstance(response, dict) or 'choices' not in response:
                    print(f"Invalid response format in batch {i}")
                    continue

                content = response['choices'][0]['message']['content']
                json_data = json_repair.loads(content)
                
                # Handle list responses and nested structures
                if isinstance(json_data, list):
                    for item in json_data:
                        if isinstance(item, dict):
                            for key, value in item.items():
                                new_key = f"prompt_{prompt_counter}"
                                combined_json[new_key] = value
                                prompt_counter += 1
                elif isinstance(json_data, dict):
                    for key, value in json_data.items():
                        new_key = f"prompt_{prompt_counter}"
                        combined_json[new_key] = value
                        prompt_counter += 1
                print(json_data)
                print(f"Added {len(json_data)} prompts from batch {i}")
                time.sleep(3)
            except Exception as e:
                print(f"Error in batch {i}: {e}")
                continue

        return combined_json

def generate_promptset(dataset_goal, dataset_type):
    search_metadata = generate_prompts(
        dataset_goal,
        seed_data_val="",
        dataset_type_val="",
        should_search_val="true"
    )
    seed_data = get_seed_data(search_metadata)
    seed_data_str = "" 
    for sample_num, data in enumerate(seed_data):  
        seed_data_str += f"The following is the {sample_num}th sample from the seed dataset:\n\n{data}\n\n"
    promptset = generate_prompts(
        dataset_goal,
        seed_data_val=seed_data_str,
        dataset_type_val=dataset_type,
        should_search_val="false"
    )
    return promptset

def get_seed_data(search_metadata):
    print(search_metadata)
    texts = []
    offset = random.randint(100, 1000)
    length = 100
    headers = {"Authorization": f"Bearer {config_values['HF_API_KEY']}"}
    if isinstance(search_metadata, list):
        search_metadata = search_metadata[0]
    dataset_types = {
        'web_data': f"https://datasets-server.huggingface.co/search?dataset=Salesforce%2Ffineweb_deduplicated&config=default&split=train&query={requests.utils.quote(search_metadata['search_term'])}&offset={offset}&length={length}",
        'educational_web_data' : f"https://datasets-server.huggingface.co/rows?dataset=HuggingFaceFW%2Ffineweb-edu&config=default&split=train&query={requests.utils.quote(search_metadata['search_term'])}&offset={offset}&length={length}",
        'code_data' : f"https://datasets-server.huggingface.co/rows?dataset=m-a-p%2FCodeFeedback-Filtered-Instruction&config=default&split=train&query={requests.utils.quote(search_metadata['search_term'])}&offset={offset}&length={length}",
        'creative_writing_data' : f"https://datasets-server.huggingface.co/rows?dataset=Lambent%2F1k-creative-writing-8kt-fineweb-edu-sample&config=default&query={requests.utils.quote(search_metadata['search_term'])}&split=train&offset={offset}&length={length}",
        'diffusiondb_data' : ''
    }
    if search_metadata['dataset_type'] in dataset_types:
        dataset_url = dataset_types[search_metadata['dataset_type']]
    else:
        dataset_url = dataset_types['web_data']
    response = requests.get(dataset_url, headers=headers, timeout=500)
    response.raise_for_status()
    for data in response.json()['rows']:
        texts.append(data['row']['text'])
    embeddings = embedding_model.encode(texts)
    embeddings_f32 = np.array(embeddings).astype('float32')
    dimension = embeddings_f32.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_f32)
    query = search_metadata['search_term']
    query_embedding = embedding_model.encode([query])
    query_embedding_f32 = query_embedding.astype('float32')
    k = 1
    _, indices = index.search(query_embedding_f32, k)
    top_texts = [texts[idx] for idx in indices[0]]
    return top_texts

def generate_chat_response(chat_history):
    message_list = [
        {
            "role": "system",
            "content": chatgen_system_prompt
        }
    ]
    message_list += chat_history
    completion = chat_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=message_list,
        temperature=0.5,
        max_tokens=4000,
        stream=False,
        response_format={"type": "json_object"},
        stop=None,
    )
    response_content = json.loads(completion.choices[0].message.content)

    if 'chat_status' in response_content and response_content['chat_status'] == "finished":
        dataset_type = response_content['dataset_type']
        dataset_goal = response_content['master_prompt']
        selected_model = response_content['selected_model']
        
        thread = threading.Thread(target=create_model, args=(dataset_type, dataset_goal))
        thread.start()
        
        return "The model creation has started and is running in the background. You will be notified once it's complete."
    else:
        return response_content.get('current_message', "No message available.")

def validate_dataset(dataset):
    """Validate the dataset structure and content"""
    return all(isinstance(v, str) for v in dataset.values())

def generate_data(dataset_type_val, dataset_goal):
    print("Generating promptset...")
    final_promptset = None
    while final_promptset == None:
        final_promptset = generate_promptset(
            dataset_goal=dataset_goal,
            dataset_type=dataset_type_val
        )
    print(f"Generated {len(final_promptset)} prompts.")
    
    if dataset_type_val == "text_only":
        search_metadata = generate_prompts(
            dataset_goal=dataset_goal,
            seed_data_val="",
            dataset_type_val=dataset_type_val,
            should_search_val="true"
        )
        seed_data = None
        while seed_data == None:
            seed_data = get_seed_data(search_metadata)
        seed_data_str = ""
        for sample_num, data in enumerate(seed_data):
            seed_data_str += f"The following is the {sample_num}th sample from the seed dataset:\n\n{data}\n\n"

        final_datagen_system_prompt = datagen_system_prompt.format(
            dataset_goal=dataset_goal,
            dataset_type=dataset_type_val,
            seed_data=seed_data_str
        )

        # Endpoint and headers for API requests
        endpoint = f"{api_url}/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config_values['AI_API_KEY']}"
        }

        final_dataset = {}
        print("\nGenerating dataset with responses...")

        # Process prompts in batches of 10
        # [list(final_promptset.items())[i:i + 100] for i in range(0, len(final_promptset), 100)]
        #prompt_batches = [list(final_promptset.items())[i:i + 100] for i in range(0, len(final_promptset), 100)]
        for prompt_num, prompt in final_promptset.items():
            print(f"Processing prompt no : {prompt_num}")
            
            # Create a batch request
            # "content": "\n".join([prompt_content for _, prompt_content in batch])
            batch_messages = [
                {
                    "role": "system",
                    "content": final_datagen_system_prompt
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            instance = {
                "model": "llama-3.3-70b",
                "messages": batch_messages,
    
                "max_tokens": 8192,
                "stream": False,

            }

            # Use make_sequential_request for retries
            try:
                response = make_sequential_request(endpoint, headers, instance, retry_count=8)
                if not isinstance(response, dict) or 'choices' not in response:
                    print(f"Invalid response format for prompt {prompt_num}")
                    continue

                content = response['choices'][0]['message']['content']
                final_response = json_repair.loads(content)

                # Handle different response formats
                if isinstance(final_response, list):
                    for item in final_response:
                        if isinstance(item, dict):
                            final_dataset.update(item)
                elif isinstance(final_response, dict):
                    final_dataset.update(final_response)

            except Exception as e:
                print(f"Error processing prompt no {prompt}: {e}")
                continue

            time.sleep(3)  # Add a small delay between batches

        print("Dataset generation completed.")

        # Validate dataset before saving
        if not validate_dataset(final_dataset):
            print("Warning: Dataset contains invalid entries")

        # Create unique filename
        os.makedirs(f"{Path.cwd()}/generated_datasets", exist_ok=True)
        unique_id = uuid.uuid4().hex[:8]
        dataset_path = f"{Path.cwd()}/generated_datasets/dataset_{unique_id}.json"

        try:
            with open(dataset_path, 'w') as f:
                json.dump(final_dataset, f, indent=4, ensure_ascii=False)
            print(f"Dataset successfully written to {dataset_path}")
        except Exception as e:
            print(f"Error writing to file: {e}")
            return None

        return [dataset_path,seed_data_str]

def create_model(dataset_type,dataset_goal,selected_model):
    dataset_path = generate_data(dataset_type,dataset_goal)
  #  finetuned_model_path = finetune_model(selected_model,dataset_path[0],dataset_type)
   # return finetuned_model_path

def qa_check(dataset_type,dataset_goal,dataset,seed_data):
    with open(dataset,'r') as dataset_file:
        current_dataset = json_repair.loads(dataset_file.read())
    similarity_threshold = 0.95
    endpoint = f"{api_url}/v1/chat/completions"
    headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config_values['AI_API_KEY']}"
        }
    final_rewritegen_system_prompt = rewritegen_system_prompt.format(
            seed_data=seed_data,
            dataset_goal = dataset_goal,
            dataset_type=dataset_type
        )
    instance = {
            "model": "llama-3.3-70b",
            "messages": [
                {"role": "system", "content": final_rewritegen_system_prompt},
                {"role": "user", "content": dataset_goal}
            ],

            "stream": False,
            "max_tokens": 8192
        }
    if dataset_type == "text_only":
        metric = BERTScore(model_name_or_path="sentence-transformers/all-MiniLM-L6-v2")
        for prompt, response in current_dataset.items():
            current_similarity_score = metric([str(response)], [str(prompt)])['f1'].item()
            current_prompt = prompt
            current_response = response
            while current_similarity_score <= similarity_threshold:
                rewrite_input = f""""
                                prompt : {current_prompt},
                                response" : {current_response},
                                similarity_score : {current_similarity_score}
                                """
                instance["messages"][0]["content"] = rewrite_input
                response = make_sequential_request(endpoint, headers, instance)
                rewritten_response = json_repair.loads(response['choices'][0]['message']['content'])
                print(rewritten_response)
                print(f"the type of returned rewritten response : {type(rewritten_response)}")
                current_similarity_score =  metric([rewritten_response['improved_response']], [rewritten_response['prompt']])['f1'].item()
                current_response = rewritten_response['improved_response']
                current_prompt = rewritten_response['prompt']   
            current_dataset[current_prompt] = current_response
        os.remove(dataset)
        unique_id = uuid.uuid4().hex[:8]
        dataset_path = f"{Path.cwd()}/generated_datasets/dataset_{unique_id}.json"
        with open(dataset_path, 'w') as f:
                json.dump(current_dataset, f, indent=4, ensure_ascii=False)
        return dataset_path
    else:
        pass