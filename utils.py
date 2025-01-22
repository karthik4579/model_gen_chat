import re
import requests
import json
from pathlib import Path
from langchain_core.prompts import PromptTemplate
from groq import Groq
from dotenv import dotenv_values
import random
import time
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google.auth import default
import uuid
import pandas as pd
import os
import threading

global final_promptset

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


# Single lock for synchronizing requests
request_lock = threading.Lock()

def make_sequential_request(endpoint, headers, instance, retry_count=3):
    with request_lock:
        for attempt in range(retry_count):
            try:
                response = requests.post(endpoint, headers=headers, json=instance, verify=False, timeout=3000)
                response.raise_for_status()
                time.sleep(2)  # Wait before next request
                return response.json()
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < retry_count - 1:
                    time.sleep(2)
                else:
                    raise

def generate_prompts(dataset_goal, seed_data_val="", dataset_type_val="", should_search_val="true"):
    final_promptgen_system_prompt = promptgen_system_prompt.format(
        seed_data=seed_data_val,
        should_search=should_search_val,
        dataset_type=dataset_type_val
    )
    
    endpoint = "http://127.0.0.1:8080/v1/chat/completions"
    headers = {
            "Content-Type": "application/json"
        }

    instance = {
        "model": "",
        "messages": [
            {"role": "system", "content": final_promptgen_system_prompt},
            {"role": "user", "content": dataset_goal}
        ],
        "temperature": 0.6,
        "top_p": 1,
        "stream": False,
        "max_tokens": 32000
    }

    if seed_data_val == "":
        search_metadata = make_sequential_request(endpoint, headers, instance)
        tag_removed_search_metadata = remove_thinking_tags(search_metadata['choices'][0]['message']['content'])
        parsed_search_metadata = json_repair.loads(tag_removed_search_metadata)
        return parsed_search_metadata
    else:
        combined_json = {}
        pattern = r"<FINAL_ANSWER>\s*(.*?)\s*</FINAL_ANSWER>"
        for i in range(10):
            try:
                content = make_sequential_request(endpoint, headers, instance)
                raw_content = content['choices'][0]['message']['content']
                match = re.search(pattern, raw_content, re.DOTALL)
                if match:
                    json_data = json.loads(match.group(1))
                    # Add prompts to combined_json
                    for key, value in json_data.items():
                        new_key = f"prompt_{len(combined_json) + 1}"
                        combined_json[new_key] = value
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
        seed_data_str += f"The following is the {sample_num}th sample from the seed dataset:\n\n{data['row']['text']}\n\n"
    promptset = generate_prompts(
        dataset_goal,
        seed_data_val=seed_data_str,
        dataset_type_val=dataset_type,
        should_search_val="false"
    )
    return promptset

def remove_thinking_tags(text):
    pattern = r'\s*<think>.*?</think>\s*'
    cleaned_text = re.sub(pattern, ' ', text, flags=re.DOTALL)
    # Clean up multiple spaces
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    return cleaned_text.strip()

def get_seed_data(search_metadata):
    offset = random.randint(0, 1000)
    length = 5
    headers = {"Authorization": f"Bearer {api_keys['HF_API_KEY']}"}
   
    if search_metadata['dataset_type'] == 'web_data':
        fineweb_url = f"https://datasets-server.huggingface.co/search?dataset=HuggingFaceFW%2Ffineweb&config=default&split=train&query={requests.utils.quote(search_metadata['search_term'])}&offset={offset}&length={length}"
        response = requests.get(fineweb_url, headers=headers,timeout=120)
        print(response.json())
        return response.json()['rows']
    elif search_metadata['dataset_type'] == 'educational_web_data':
        fineweb_edu_url = f"https://datasets-server.huggingface.co/rows?dataset=HuggingFaceFW%2Ffineweb-edu&config=default&split=train&query={requests.utils.quote(search_metadata['search_term'])}&offset={offset}&length={length}"
        response = requests.get(fineweb_edu_url, headers=headers,timeout=120)
        return response.json()['rows']
    
    elif search_metadata['dataset_type'] == 'code_data':
        code_feedback_url = f"https://datasets-server.huggingface.co/rows?dataset=m-a-p%2FCodeFeedback-Filtered-Instruction&config=default&split=train&query={requests.utils.quote(search_metadata['search_term'])}&offset={offset}&length={length}"
        response = requests.get(code_feedback_url, headers=headers,timeout=120)
        return response.json()['rows']
    
    elif search_metadata['dataset_type'] == 'creative_writing_data':
        creative_writing_url = f"https://datasets-server.huggingface.co/rows?dataset=Lambent%2F1k-creative-writing-8kt-fineweb-edu-sample&config=default&query={requests.utils.quote(search_metadata['search_term'])}&split=train&offset={offset}&length={length}"
        response = requests.get(creative_writing_url, headers=headers,timeout=120)
        return response.json()['rows']
    
    elif search_metadata['dataset_type'] == 'diffusiondb_data':
        pass
    else:
        pass

def generate_chat_response(chat_history):
    message_list = [
        {
            "role": "system",
            "content": chatgen_system_prompt
        }
    ]
    message_list += chat_history
    completion = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
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
        
        # Run generate_data in a separate thread
        thread = threading.Thread(target=generate_data, args=(dataset_type, dataset_goal))
        thread.start()
        
        return "The model creation has started and is running in the background. You will be notified once it's complete."
    else:
        return response_content.get('current_message', "No message available.")


def generate_data(dataset_type_val, dataset_goal):
    final_promptset = generate_promptset(
        dataset_goal=dataset_goal,
        dataset_type=dataset_type_val
    )
    
    # Get seed data
    search_metadata = generate_prompts(dataset_goal, seed_data_val="", dataset_type_val=dataset_type_val, should_search_val="true")
    seed_data = get_seed_data(search_metadata)
    seed_data_str = ""
    # Limit to 4 samples as requested
    for sample_num, data in enumerate(seed_data[:4]):
        seed_data_str += f"The following is the {sample_num}th sample from the seed dataset:\n\n{data['row']['text']}\n\n"

    final_datagen_system_prompt = datagen_system_prompt.format(
        dataset_goal=dataset_goal,
        correction_status="false",
        dataset_type=dataset_type_val,
        seed_data=seed_data_str
    )
    
    # Use the self-hosted endpoint
    endpoint = "http://127.0.0.1:8080/v1/chat/completions"
    headers = {
        "Content-Type": "application/json"
    }

    final_dataset = {}
    
    pattern = r"<FINAL_ANSWER>\s*(.*?)\s*</FINAL_ANSWER>"
    
    for prompt_num, (prompt_key, prompt_content) in enumerate(final_promptset.items(), 1):
        instance = {
            "model": "",
            "messages": [
                {"role": "system", "content": final_datagen_system_prompt},
                {"role": "user", "content": prompt_content}
            ],
            "temperature": 0.6,
            "max_tokens": 32000,
            "stream": False
        }
        
        try:
            response = requests.post(endpoint, headers=headers, json=instance, verify=False,timeout=3000)
            response.raise_for_status()
            content = response.json()
            full_content = content['choices'][0]['message']['content']
            try:
                match = re.search(pattern, full_content, re.DOTALL)
                if match:
                    final_response = json.loads(match.group(1))
                    final_dataset[f"prompt_{prompt_num}"] = final_response
                else:
                    final_dataset[f"prompt_{prompt_num}"] = {"error": "No final answer found in response"}
            except:
                final_dataset[f"prompt_{prompt_num}"] = {"error": "Invalid JSON in response"}
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error for prompt_{prompt_num}: {http_err}")
            final_dataset[f"prompt_{prompt_num}"] = {"error": f"HTTP error: {http_err}"}
        except Exception as err:
            print(f"Error processing prompt_{prompt_num}: {err}")
            final_dataset[f"prompt_{prompt_num}"] = {"error": f"An error occurred: {err}"}
        
        time.sleep(1)
    
    # Write the final_dataset to dataset.json
    dataset_path = f"{Path.cwd()}/generated_datasets/dataset.json"
    try:
        with open(dataset_path, 'w') as f:
            json.dump(final_dataset, f, indent=4)
        print("Dataset successfully written to dataset.json")
    except Exception as e:
        print(f"Error writing to file: {e}")

    return final_dataset


def create_directory(uuid_name):
    os.makedirs(uuid_name, exist_ok=True)
    return uuid_name


def save_as_json(data, directory, filename):
    with open(os.path.join(directory, f"{filename}.json"), 'w') as f:
        json.dump(data, f, indent=4)
