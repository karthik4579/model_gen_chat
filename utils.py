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

model = SentenceTransformer('NovaSearch/stella_en_400M_v5',trust_remote_code=True)

api_keys = dotenv_values(f"{Path.cwd()}/config.env")
client = Groq(api_key=api_keys["GROQ_API_KEY"])

with open(f"{Path.cwd()}/prompts/prompt_gen.txt") as promptgen_prompt:
    raw_promptgen_system_prompt = promptgen_prompt.read()
with open(f"{Path.cwd()}/prompts/chat_gen.txt") as chatgen_prompt:
    chatgen_system_prompt = chatgen_prompt.read()
with open(f"{Path.cwd()}/prompts/data_gen.txt") as datagen_prompt:
    raw_datagen_system_prompt = datagen_prompt.read()

promptgen_system_prompt = PromptTemplate.from_template(raw_promptgen_system_prompt)
datagen_system_prompt = PromptTemplate.from_template(raw_datagen_system_prompt)

request_lock = threading.Lock()

def make_sequential_request(endpoint, headers, instance, retry_count=5):
    with request_lock:
        for attempt in range(retry_count):
            try:
                response = requests.post(endpoint, headers=headers, json=instance, timeout=3000)
                response.raise_for_status()
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

    endpoint = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_keys['GROQ_API_KEY']}"
    }

    instance = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": final_promptgen_system_prompt},
            {"role": "user", "content": dataset_goal}
        ],
        "temperature": 0.5,
        "stream": False,
        "max_tokens": 32000, 
        "stop": None
    }

    if seed_data_val == "":
        try:
            search_metadata = make_sequential_request(endpoint, headers, instance)
            if isinstance(search_metadata, dict) and 'choices' in search_metadata:
                content = search_metadata['choices'][0]['message']['content']
                return json_repair.loads(content)
            raise ValueError("Invalid response format")
        except Exception as e:
            print(f"Error in search metadata generation: {e}")
            raise
    else:
        combined_json = {}
        prompt_counter = 1  # Track total prompts across batches

        for i in range(5):
            try:
                response = make_sequential_request(endpoint, headers, instance)
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
    texts = []
    offset = random.randint(100, 1000)
    length = 100
    headers = {"Authorization": f"Bearer {api_keys['HF_API_KEY']}"}
    if isinstance(search_metadata, list):
        search_metadata = search_metadata[0]
    print(search_metadata)
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
    embeddings = model.encode(texts)
    embeddings_f32 = np.array(embeddings).astype('float32')
    dimension = embeddings_f32.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_f32)
    query = search_metadata['search_term']
    query_embedding = model.encode([query])
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
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=message_list,
        temperature=0.5,
        max_tokens=4000,
        top_p=1,
        stream=False,
        response_format={"type": "json_object"},
        stop=None,
    )
    response_content = json.loads(completion.choices[0].message.content)

    if 'chat_status' in response_content and response_content['chat_status'] == "finished":
        dataset_type = response_content['dataset_type']
        dataset_goal = response_content['master_prompt']
        
        thread = threading.Thread(target=generate_data, args=(dataset_type, dataset_goal))
        thread.start()
        
        return "The model creation has started and is running in the background. You will be notified once it's complete."
    else:
        return response_content.get('current_message', "No message available.")

def validate_dataset(dataset):
    """Validate the dataset structure and content"""
    return all(isinstance(v, str) for v in dataset.values())

def generate_data(dataset_type_val, dataset_goal):
    print("Generating promptset...")
    final_promptset = generate_promptset(
        dataset_goal=dataset_goal,
        dataset_type=dataset_type_val
    )
    print(f"Generated {len(final_promptset)} prompts.")
    
    # Save the generated promptset to a file
    dataset_path = f"{Path.cwd()}/generated_datasets/prompts.json"
    try:
        with open(dataset_path, 'w') as f:
            json.dump(final_promptset, f, indent=4, ensure_ascii=False)
        print("Promptset generation completed.")
    except Exception as e:
        print(f"Error saving promptset: {e}")
        return None

    # Prepare system prompt for data generation
    search_metadata = generate_prompts(
        dataset_goal=dataset_goal,
        seed_data_val="",
        dataset_type_val=dataset_type_val,
        should_search_val="true"
    )
    seed_data = get_seed_data(search_metadata)
    seed_data_str = ""
    for sample_num, data in enumerate(seed_data):
        seed_data_str += f"The following is the {sample_num}th sample from the seed dataset:\n\n{data}\n\n"

    final_datagen_system_prompt = datagen_system_prompt.format(
        dataset_goal=dataset_goal,
        correction_status="false",
        dataset_type=dataset_type_val,
        seed_data=seed_data_str
    )

    # Endpoint and headers for API requests
    endpoint = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_keys['GROQ_API_KEY']}"
    }

    final_dataset = {}
    print("\nGenerating dataset with responses...")

    # Process prompts in batches of 10
    prompt_batches = [list(final_promptset.items())[i:i + 50] for i in range(0, len(final_promptset), 30)]
    for batch_num, batch in enumerate(prompt_batches):
        print(f"Processing batch {batch_num + 1} of {len(prompt_batches)}")
        
        # Create a batch request
        batch_messages = [
            {
                "role": "system",
                "content": final_datagen_system_prompt
            },
            {
                "role": "user",
                "content": "\n".join([prompt_content for _, prompt_content in batch])
            }
        ]
        instance = {
            "model": "llama-3.3-70b-versatile",
            "messages": batch_messages,
            "temperature": 0.5,
            "max_tokens": 32000,
            "stream": False,
            "stop": None
        }

        # Use make_sequential_request for retries
        try:
            response = make_sequential_request(endpoint, headers, instance, retry_count=5)
            if not isinstance(response, dict) or 'choices' not in response:
                print(f"Invalid response format for batch {batch_num + 1}")
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
            print(f"Error processing batch {batch_num + 1}: {e}")
            continue

        time.sleep(10)  # Add a small delay between batches

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

    return dataset_path

def create_directory(uuid_name):
    os.makedirs(uuid_name, exist_ok=True)
    return uuid_name

def save_as_json(data, directory, filename):
    with open(os.path.join(directory, f"{filename}.json"), 'w') as f:
        json.dump(data, f, indent=4)