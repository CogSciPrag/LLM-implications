import os
from pprint import pprint

import openai
from langchain.document_loaders import ImageCaptionLoader
from langchain.indexes import VectorstoreIndexCreator

from dotenv import load_dotenv
from PIL import Image
import requests

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def generate_captions(images=""):
    """
    Wrapper calling the image captioning end point of LangChain with default captioning model.
    Uses a vector storage as an intermediate step for running the captioning task on the captioner ouput.

    Parameters
    ----------
    images : list
        List of image URLs (strings).
    Returns:
    --------
    response: str
        Formatted Response of the image captioning model.
    """
    # load images and create captions
    # by default, the model Salesforce/blip-image-captioning-base is used
    loader = ImageCaptionLoader(path_images=images)
    # print("Raw image caption documents: \n", loader.load())
    # create vector store (chroma DB) with images + captions
    # by default, vectors (=embeddings) are created with OpenAI embeddings
    index = VectorstoreIndexCreator().from_loaders([loader])
    # based on these embeddings, the response to the query is computed
    query = "What do the images show?"
    response = index.query(query)
    
    return response
    

if __name__ == "__main__":
    
    images = [
        "http://images.cocodataset.org/val2014/COCO_val2014_000000391895.jpg",
        "http://images.cocodataset.org/val2014/COCO_val2014_000000079841.jpg",
    ]

    r = generate_captions(images)    

    pprint(r)

    # plot the images 
    
    im1 = Image.open(requests.get(images[0], stream=True).raw).convert('RGB')

    im2 = Image.open(requests.get(images[1], stream=True).raw).convert('RGB')

    im1.show()
    im2.show()