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
import threading
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import uuid
import time
#from fine_tuning_utils import finetune_model
from sentence_transformers import SentenceTransformer
from torchmetrics.text.bert import BERTScore
import concurrent.futures

embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

config_values = dotenv_values(f"{Path.cwd()}/config.env")
api_url = config_values['AI_API_URL']
client = Groq(api_key=config_values["AI_API_KEY"])
chat_client = Groq(api_key=config_values["AI_API_KEY_CHAT"])
main_model_name = config_values["MAIN_AI_MODEL_NAME"]

with open(f"{Path.cwd()}/prompts/prompt_gen.txt") as promptgen_prompt:
    raw_promptgen_system_prompt = promptgen_prompt.read()
with open(f"{Path.cwd()}/prompts/chat_gen.txt") as chatgen_prompt:
    chatgen_system_prompt = chatgen_prompt.read()
with open(f"{Path.cwd()}/prompts/data_gen.txt") as datagen_prompt:
    raw_datagen_system_prompt = datagen_prompt.read()
with open(f"{Path.cwd()}/prompts/rewrite_gen_text.txt") as rewritegen_prompt:
    raw_rewrite_system_prompt = rewritegen_prompt.read()
with open(f"{Path.cwd()}/prompts/llm_judge_prompt.txt") as llm_judge_prompt:
    llm_judge_prompt = llm_judge_prompt.read()

promptgen_system_prompt = PromptTemplate.from_template(raw_promptgen_system_prompt)
datagen_system_prompt = PromptTemplate.from_template(raw_datagen_system_prompt)
rewritegen_system_prompt = PromptTemplate.from_template(raw_rewrite_system_prompt)

def make_sequential_request(endpoint, header, instance, retry_count=5):
    for attempt in range(retry_count):
            try:
                response = requests.post(endpoint, headers=header, json=instance, timeout=7000)
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

    endpoint = f"{api_url}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config_values['AI_API_KEY']}"
    }

    instance = {
        "model": main_model_name,
        "messages": [
            {"role": "system", "content": final_promptgen_system_prompt},
            {"role": "user", "content": f"{dataset_goal}"}
        ],
        "stream": False,
        "temperature": 0.6,
        "max_tokens": 64000,
        'provider': {
      'order': ["DeepInfra"]
    }
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
        prompt_counter = 1

        def process_batch(batch_num):
            try:
                response = make_sequential_request(endpoint, headers, instance)
                print(response)
                if not isinstance(response, dict) or 'choices' not in response:
                    print(f"Invalid response format in batch {batch_num}")
                    return None
                content = response['choices'][0]['message']['content']
                json_data = json_repair.loads(content)
                return json_data
            except Exception as e:
                print(f"Error in batch {batch_num}: {e}")
                return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=40) as executor:
            futures = {executor.submit(process_batch, i): i for i in range(1,31)}
            for future in concurrent.futures.as_completed(futures):
                json_data = future.result()
                if json_data is None:
                    continue
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
                print(f"Added prompts from batch {futures[future]}")
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
        'educational_web_data' : f"https://datasets-server.huggingface.co/rows?dataset=skymizer%2Ffineweb-edu-dedup-45B&config=default&split=train&query={requests.utils.quote(search_metadata['search_term'])}&offset={offset}&length={length}",
        'code_data' : f"https://datasets-server.huggingface.co/rows?dataset=m-a-p%2FCodeFeedback-Filtered-Instruction&config=default&split=train&query={requests.utils.quote(search_metadata['search_term'])}&offset={offset}&length={length}",
        'creative_writing_data' : f"https://datasets-server.huggingface.co/rows?dataset=Lambent%2F1k-creative-writing-8kt-fineweb-edu-sample&config=default&query={requests.utils.quote(search_metadata['search_term'])}&split=train&offset={offset}&length={length}",
    }
    if search_metadata['dataset_type'] in dataset_types:
        dataset_url = dataset_types[search_metadata['dataset_type']]
    else:
        dataset_url = dataset_types['web_data']
    
    try:
        print(f"searching from {search_metadata['dataset_type']} so NO fail over")
        response = requests.get(dataset_url, headers=headers, timeout=1000)
    except:
        print("switched to searching from default fineweb dataset")
        response = requests.get(dataset_types['web_data'], headers=headers, timeout=1000)
        
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
        temperature=0.6,
        max_tokens=4000,
        stream=False,
        response_format={"type": "json_object"},
        stop=None
    )
    response_content = json.loads(completion.choices[0].message.content)
    print(response_content)

    if 'chat_status' in response_content and response_content['chat_status'] == "finished":
        dataset_type = response_content['dataset_type']
        dataset_goal = response_content['master_prompt']
        selected_model = response_content['selected_model']
        
        thread = threading.Thread(target=create_model, args=(dataset_type, dataset_goal,selected_model))
        thread.start()
        return "The model creation has started and is running in the background. You will be notified once it's complete."
    else:
        return response_content.get('current_message', "No message available.")

def generate_data(dataset_type_val, dataset_goal):
    print("Generating promptset...")
    final_promptset = None
    while final_promptset is None:
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
        while seed_data is None:
            seed_data = get_seed_data(search_metadata)
        seed_data_str = ""
        for sample_num, data in enumerate(seed_data):
            seed_data_str += f"The following is the {sample_num}th sample from the seed dataset:\n\n{data}\n\n"

        system_prompt = datagen_system_prompt.format(
            dataset_goal=dataset_goal,
            dataset_type=dataset_type_val,
            seed_data=seed_data_str
        )
        endpoint = f"{api_url}/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config_values['AI_API_KEY']}"
        }
        final_raw_dataset = {}

        def process_prompt(prompt_num, prompt):
            print(f"Processing prompt no : {prompt_num}")
            batch_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{prompt}"}
            ]
            instance = {
                "model": main_model_name,
                "messages": batch_messages,
                "temperature": 0.5,
                "max_tokens": 64000,
                "stream": False,
                'provider': {
      'order': [ "DeepInfra"]
    }
            }
            try:
                response = make_sequential_request(endpoint, headers, instance, retry_count=8)
                if not isinstance(response, dict) or 'choices' not in response:
                    print(f"Invalid response format for prompt {prompt_num}")
                    return {}
                content = response['choices'][0]['message']['content']
                final_response = json_repair.loads(content)
                result = {}
                if isinstance(final_response, list):
                    for item in final_response:
                        if isinstance(item, dict):
                            result.update(item)
                elif isinstance(final_response, dict):
                    result.update(final_response)
                return result
            except Exception as e:
                print(f"Error processing prompt no {prompt_num}: {e}")
                return {}

        print("\nGenerating dataset with responses...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=40) as executor:
            futures = {
                executor.submit(process_prompt, prompt_num, prompt): prompt_num
                for prompt_num, prompt in final_promptset.items()
            }
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                final_raw_dataset.update(res)

        print("Dataset generation completed.")

        return [final_raw_dataset, seed_data_str]

def create_model(dataset_type,dataset_goal,selected_model):
    dataset_path = generate_data(dataset_type,dataset_goal)
    #finetuned_model_path = finetune_model(selected_model,dataset_path[0],dataset_type)
    #return finetuned_model_path

def qa_check(dataset_type,dataset_goal,dataset,seed_data):
    current_dataset = dataset
    similarity_threshold = 80
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
    if dataset_type == "text_only":
        metric = BERTScore(model_name_or_path="answerdotai/ModernBERT-base")
        for prompt, response in current_dataset.items():
            score = metric([str(response)], [str(prompt)])['f1'].item()
            current_similarity_score = round(score,2) * 100
            current_prompt = prompt
            current_response = response
            if 70 <= current_similarity_score <= similarity_threshold:
                 current_dataset[current_prompt] = current_response      
            else:
                input_eval_prompt = llm_judge_prompt.format(dataset_goal=dataset_goal,user_input=current_prompt,assistant_response=current_response)
                eval_instance = {
                "model": "",
                "messages": [
                    {"role": "user", "content": input_eval_prompt}
                ],
                "temperature": 0.5,
                "stream": False,
                "max_tokens": 16000
            }
                eval_response = requests.post(url="https://kit-polished-terminally.ngrok-free.app/v1/chat/completions",json=eval_instance,timeout=6000).json()['choices'][0]['message']['content']
                evals = json_repair.loads(eval_response)
                print(evals)
                current_critique = evals["reasoning"]
                eval_result = evals["result"]
                   
                if eval_result >= 4:
                    print("greater than 4 so adding to dataset ...")
                    current_dataset[current_prompt] = current_response
                else:
                    print("less than 4 judge response so entering loop ...")
                    current_eval_result = eval_result
                    while current_eval_result <= 4:
                        final_rewritegen_system_prompt = rewritegen_system_prompt.format(dataset_goal=dataset_goal)
                        rewrite_input = f"""
                                        prompt : {current_prompt}
                                        response : {current_response}
                                        critique : {current_critique}
                                        current_score : {current_eval_result}
                                        """
                        instance = {
                    "model": main_model_name,
                    "messages": [
                        {"role": "system", "content": final_rewritegen_system_prompt},
                        {"role": "user", "content": rewrite_input}
                    ],
                    "stream": False,
                    "max_tokens": 32000,
                    "temperature": 0.6,
                    'provider': {
      'order': [ "DeepInfra"]
    }
                    }   
                        response = make_sequential_request(endpoint, headers, instance)
                        rewritten_response = json_repair.loads(response['choices'][0]['message']['content'])
                        print(rewritten_response)
                        current_response = rewritten_response['improved_response']
                        current_prompt = rewritten_response['prompt']
    
                        input_eval_prompt = llm_judge_prompt.format(dataset_goal=dataset_goal,user_input=current_prompt,assistant_response=current_response)
                        eval_instance = {
                        "model": "",
                        "messages": [
                            {"role": "user", "content": input_eval_prompt}
                        ],
                        "temperature": 0.5,
                        "stream": False,
                        "max_tokens": 8192
                    }
                    eval_response = requests.post(url="https://kit-polished-terminally.ngrok-free.app/v1/chat/completions",json=eval_instance,timeout=600).json()['choices'][0]['message']['content']
                    evals = json_repair.loads(eval_response)
                    print(evals)
                    print(type(evals))
                    if evals["reasoning"] and evals["result"]:
                        current_critique = evals["reasoning"]
                        current_eval_result = evals["result"]
            
                    current_dataset[current_prompt] = current_response
        
        dataset_path = f"{Path.cwd()}/generated_datasets/{uuid.uuid4().hex[:12]}"
        temp_final_dataset = list(current_dataset.items())
        train_samples = temp_final_dataset[:300]
        test_samples = temp_final_dataset[300:]
        try:
            os.makedirs(dataset_path, exist_ok=True)
            with open(dataset_path + "/train.json", 'w') as train_file:
                    json.dump(dict(train_samples), train_file, indent=4, ensure_ascii=False)
                    print(f"train set successfully written to {dataset_path + '/train.json'}")
                
            with open(dataset_path + "/test.json", 'w') as test_file:
                    json.dump(dict(test_samples), test_file, indent=4, ensure_ascii=False)
                    print(f"test set successfully written to {dataset_path + '/test.json'}")
        except Exception as e:
            print(f"Error writing dataset to disk: {e}")
        
        return dataset_path
    else:
        pass