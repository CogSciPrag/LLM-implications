import json
import requests
from dotenv import load_dotenv
import os

load_dotenv()
API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
# API_URL = "https://api-inference.huggingface.co/models/OpenAssistant/oasst-sft-1-pythia-12b-epoch-3.5"
# API_URL = "https://api-inference.huggingface.co/models/OpenAssistant/oasst-sft-1-pythia-12b"
API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-xxl"
headers = {"Authorization": f"Bearer {API_TOKEN}",  "Content-Type": "application/json"}
def query(payload):
    data = json.dumps(payload)
    print("data input ", data)
    response = requests.request("POST", API_URL, headers=headers, data=data)
    print("response ", response)
    return json.loads(response.content.decode("utf-8"))

data = query(
    {"inputs": "<|prompter|>Imagine you are a scientist. Your paper just go accepted into a top tier conference. On a scale of 1 to 10, how important is this event for you? return the number only.<|endoftext|><|assistant|>",
     "max_new_tokens": 250,
     "temperature": 10,
     })
print("Response: ", data)