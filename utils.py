import os
import json
import time
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

model_list = {
    'gpt-4o-2024-08-06' : OpenAI(base_url = os.getenv('BASE_URL_GPT'),
                   api_key = os.getenv('API_KEY_GPT')),
    'gpt-4o-mini' : OpenAI(base_url = os.getenv('BASE_URL_GPT'),
                   api_key = os.getenv('API_KEY_GPT')),
    'DeepSeekV3' : OpenAI(base_url = os.getenv('ASE_URL_DEEPSEEK'),
                   api_key = os.getenv('API_KEY_DEEPSEEK')),
}

def get_model(model_name):
    try:
        model = model_list.get(model_name)
    except Exception as e:
        print(f"no such model:{e}")

    if not model:
        raise ValueError(f"{model_name} is not available.")
    
    return model

def invoke_model(model, prompt, model_name, temperature = 0):
    messages = [{"role": "user", "content": prompt}]
    for _ in range(3):
        try:
            completion = model.chat.completions.create(
                model = model_name,
                messages = messages,
                temperature = temperature
            )
            output = completion.choices[0].message.content
            return output
        except Exception as e:
            print(f"{model_name} doesn't work: {e}")
            time.sleep(2)
            continue
    return "Invoke Model Error!"

def invoke_model_with_system_prompt(model, system_prompt, prompt, model_name, temperature = 0):
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}]
    for _ in range(3):
        try:
            completion = model.chat.completions.create(
                model = model_name,
                messages = messages,
                temperature = temperature
            )
            output = completion.choices[0].message.content
            return output
        except Exception as e:
            print(f"{model_name} doesn't work: {e}")
            time.sleep(2)
            continue
    return "Invoke Model With System Prompt Error!"

def invoke_multi_model(model, message, model_name, temperature = 0):
    messages = message
    for _ in range(3):
        try:
            completion = model.chat.completions.create(
                model = model_name,
                messages = messages,
                temperature = temperature
            )
            output = completion.choices[0].message.content
            return output
        except Exception as e:
            print(f"{model_name} doesn't work: {e}")
            time.sleep(2)
            continue
    return "Invoke Multi Model Error!"



def import_prompt(prompt_path):
    with open(prompt_path,'r', encoding='utf-8') as file:
        prompt = file.read()
    return prompt

def get_max_filename(data_path):
        files = os.listdir(data_path)
        files = [f for f in files if os.path.isfile(os.path.join(data_path, f))]
    
        if not files:
            return None
        
        return max(files)
    
def read_json(data_path):
    with open(data_path, mode='r', encoding='utf-8') as file:
        return json.load(file)
    
def save_json(date, data_path):
    current_time = datetime.now().strftime("%Y%m%d_%H%M")
    file_name = f"result_{current_time}.json"
    file_path = data_path + "/" + file_name
    with open(file_path, mode='w', encoding='utf-8') as file:
        json.dump(date, file, ensure_ascii=False, indent=4)

