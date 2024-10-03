import concurrent.futures
import requests
import json
import re
from pathlib import Path
from langchain_core.prompts import PromptTemplate
from groq import Groq
from dotenv import dotenv_values
import random
import time
import json_repair

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


def generate_prompts(dataset_goal, seed_data_val="", dataset_type_val="", should_search_val="true"):
    final_promptgen_system_prompt = promptgen_system_prompt.format(seed_data=seed_data_val,should_search=should_search_val,dataset_type=dataset_type_val)
    
    url = "https://api.deepinfra.com/v1/openai/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "X-Deepinfra-Source": "web-page",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
    }
    
    payload = {
        "model": "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "messages": [
            {"role": "system", "content": final_promptgen_system_prompt},
            {"role": "user", "content": dataset_goal}
        ],
        "stream": True
    }

    all_responses = []
    combined_json = {}
    final_json_data = {}
    
    if seed_data_val == "":
        response = requests.post(url, headers=headers, json=payload, stream=True)
        full_content = ""
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith("data: "):
                    try:
                        json_data = json.loads(line[6:])
                        if 'choices' in json_data and json_data['choices']:
                            delta = json_data['choices'][0].get('delta', {})
                            if 'content' in delta:
                                full_content += delta['content']
                    except json.JSONDecodeError:
                        continue
        
        pattern = r'<start_json>(.*?)<end_json>'
        raw_json = re.search(pattern, full_content, re.DOTALL)
        if raw_json:
            final_json_data = json_repair.loads(raw_json.group(1))
        print(final_json_data)
        return final_json_data
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(requests.post, url, headers=headers, json=payload, stream=True) for _ in range(3)]
            batch_responses = [future.result() for future in futures]
            
        for response in batch_responses:
            full_content = ""
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith("data: "):
                        try:
                            json_data = json.loads(line[6:])
                            if 'choices' in json_data and json_data['choices']:
                                delta = json_data['choices'][0].get('delta', {})
                                if 'content' in delta:
                                    full_content += delta['content']
                        except json.JSONDecodeError:
                            continue
            all_responses.append(full_content)
    
        for sample in all_responses:
            try:
                pattern = r'<start_json>(.*?)<end_json>'
                actual_json = re.search(pattern, sample, re.DOTALL)
                if actual_json:
                    if combined_json == {}: 
                        json_data = json_repair.loads(actual_json.group(1))
                        combined_json.update(json_data)
                    else:
                        json_data = json_repair.loads(actual_json.group(1))
                        combined_json.update({f"prompt_{len(combined_json) + 1}": json_data[val] for val in list(json_data.keys())})
            except (KeyError, json.JSONDecodeError, AttributeError) as e:
                print(f"Error processing response {sample}: {e}")
                
        with open(f"{Path.cwd()}/prompt.json","w") as json_file:
            json_file.write(json.dumps(combined_json))
        return combined_json

def generate_promptset(dataset_goal,dataset_type):
    search_metadata = generate_prompts(dataset_goal, seed_data_val="", dataset_type_val="", should_search_val="true")
    seed_data = get_seed_data(search_metadata)
    seed_data_str = "" 
    for sample_num,data in enumerate(seed_data):  
        seed_data_str += f"The following is the {sample_num}th sample from the seed dataset:\n\n{data['row']['text']}\n\n"
    promptset = generate_prompts(dataset_goal, seed_data_val=seed_data_str, dataset_type_val=dataset_type, should_search_val="false")
    final_promptset = promptset
    return promptset

def get_seed_data(search_metadata):
    offset = random.randint(0,1000)
    headers = {"Authorization": f"Bearer {api_keys['HF_API_KEY']}"}
    if search_metadata['dataset_type'] == 'web_data':
        fineweb_url = f"https://datasets-server.huggingface.co/search?dataset=nampdn-ai%2Fmini-fineweb&config=default&split=train&query={requests.utils.quote(search_metadata['search_term'])}&offset={offset}&length=1"
        response = requests.get(fineweb_url, headers=headers)
        return response.json()['rows']
    if search_metadata['dataset_type'] == 'educational_web_data':
        fineweb_edu_url = f"https://datasets-server.huggingface.co/rows?dataset=HuggingFaceFW%2Ffineweb-edu&config=default&split=train&query={requests.utils.quote(search_metadata['search_term'])}&offset={offset}&length=1"
        response = requests.get(fineweb_edu_url, headers=headers)
        return response.json()['rows']
    
    if search_metadata['dataset_type'] == 'code_data':
        code_feedback_url = f"https://datasets-server.huggingface.co/rows?dataset=m-a-p%2FCodeFeedback-Filtered-Instruction&config=default&split=train&query={requests.utils.quote(search_metadata['search_term'])}&offset={offset}&length=1"
        response = requests.get(code_feedback_url, headers=headers)
        return response.json()['rows']
    
    if search_metadata['dataset_type'] == 'creative_writing_data':
        creative_writing_url = f"https://datasets-server.huggingface.co/rows?dataset=Lambent%2F1k-creative-writing-8kt-fineweb-edu-sample&config=default&query={requests.utils.quote(search_metadata['search_term'])}&split=train&offset={offset}&length=1"
        response = requests.get(creative_writing_url, headers=headers)
        return response.json()['rows']
        
    if search_metadata['dataset_type'] == 'diffusiondb_data':
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
    temperature=0.6,
    max_tokens=5000,
    top_p=1,
    stream=False,
    response_format={"type": "json_object"},
    stop=None,
    )
    response = json.loads(completion.choices[0].message.content)
    '''
    if 'chat_status' in response and response['chat_status'] == "finished":
        print(response)
    '''
    return json.loads(completion.choices[0].message.content)['current_message']

def generate_data(dataset_type_val, dataset_goal):
    final_promptset = generate_promptset(dataset_goal=dataset_goal, dataset_type=dataset_type_val)
    final_datagen_system_prompt = datagen_system_prompt.format(dataset_goal=dataset_goal, correction_status="false", dataset_type=dataset_type_val, seed_data=final_promptset)
    
    url = "https://api.deepinfra.com/v1/openai/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "X-Deepinfra-Source": "web-page",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
    }
    
    payload = {
        "model": "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "messages": [
            {"role": "system", "content": final_datagen_system_prompt},
            {"role": "user", "content": f"{final_promptset}"}
        ],
        "stream": True,
        "max_tokens": 70000
    }
    
    response = requests.post(url, headers=headers, json=payload, stream=True)
    
    full_content = ""
    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8')
            if line.startswith("data: "):
                try:
                    json_data = json.loads(line[6:])
                    if 'choices' in json_data and json_data['choices']:
                        delta = json_data['choices'][0].get('delta', {})
                        if 'content' in delta:
                            full_content += delta['content']
                except json.JSONDecodeError:
                    continue

    # Apply the regex parsing to the full_content
    pattern = r'<start_json>(.*?)<end_json>'
    raw_json = re.search(pattern, full_content, re.DOTALL)
    if raw_json:
        final_response = json_repair.loads(raw_json.group(1))
    else:
        final_response = {"error": "No JSON data found in the response"}

    print(final_response)
    
    # Uncomment the following lines if you want to write the response to a file
    # with open(f"{Path.cwd()}/data.json", "w") as json_file:
    #     json_file.write(json.dumps(final_response))

    return final_response