import json
import requests
from dotenv import load_dotenv
import os

load_dotenv()
API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

# below are URLs to models that we have briefly tested
# uncomment respectively to test
# API_URL = "https://api-inference.huggingface.co/models/OpenAssistant/oasst-sft-1-pythia-12b-epoch-3.5" (might time out)
# API_URL = "https://api-inference.huggingface.co/models/OpenAssistant/oasst-sft-1-pythia-12b"
API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-xxl"

headers = {"Authorization": f"Bearer {API_TOKEN}",  "Content-Type": "application/json"}

def query(payload):
    """
    Helper for querying the API.
    """
    data = json.dumps(payload)
    
    response = requests.request("POST", API_URL, headers=headers, data=data)
    
    return json.loads(response.content.decode("utf-8"))

# OpenAssistant provides models fine-tuned for chat
# therefore, based on their docs on Huggingface, 
# the special tokens you can see in the input string are supposed to be used.
data_oassist = query(
    {"inputs": "<|prompter|>Imagine you are a scientist. Your paper just go accepted into a top tier conference. On a scale of 1 to 10, how important is this event for you? return the number only.<|endoftext|><|assistant|>",
     "max_new_tokens": 250,
     "temperature": 10,
     }
)

data_flan_t5 = query(
    {"inputs": "Imagine you are a scientist. Your paper just go accepted into a top tier conference. On a scale of 1 to 10, how important is this event for you? return the number only.",
     "max_new_tokens": 250,
     "temperature": 10,
    }

)
print("Response form OpenAssistant model: ", data_oassist)
print("Response form Flan-T5 model: ", data_flan_t5)
