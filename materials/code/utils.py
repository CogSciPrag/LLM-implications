from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain import HuggingFaceHub
import os

def init_model(model_name, **kwargs):
    """
    Helper for initializing different models as the LLM backbone.
    
    Parameters:
    ----------
    model_name: str
        Model to be used. Has to be a model known to langchain.

    Returns:
    --------
    model: langchain.model
        Initialized model.
    """

    if model_name in ["gpt-3.5-turbo", "gpt-4"]:
        # load model
        model = ChatOpenAI(
            model_name=model_name,
            **kwargs
        )
    elif any([s in model_name for s in ["davinci", "ada", "babbage", "curie"]]):
        model = OpenAI(
            model=model_name,
            **kwargs
        )
    elif "flan-t5" in model_name:
        try:
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.environ["HUGGINGFACE_API_TOKEN"]
        except:
            try:
                assert os.environ["HUGGINGFACEHUB_API_TOKEN"] != "" 
            except:
                print("Huggingface API token is missing! Add the token to your env file with they key HUGGINGFACEHUB_API_TOKEN")
        
        model = HuggingFaceHub(
            repo_id=model_name, 
            model_kwargs=kwargs
        )
    else:
        raise ValueError(f"Unknown or incorrect model name {model_name}.\
                         See https://python.langchain.com/en/latest/modules/models/llms/integrations.html \
                         for a list of available models.")
    return model