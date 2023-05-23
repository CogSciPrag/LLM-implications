import os
import pandas as pd
from dotenv import load_dotenv

import numpy as np
from langchain import PromptTemplate, FewShotPromptTemplate
from langchain.prompts.example_selector.base import BaseExampleSelector
from langchain.chains.llm import LLMChain
from langchain.chains import SequentialChain, TransformChain
import openai

from utils import init_model
import re
from pprint import pprint

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class RandomExampleSelector(BaseExampleSelector):
    
    def __init__(self, examples: pd.DataFrame, num_facts=1):
        self.examples = examples
        self.num_facts = num_facts
       
        self.keys = ["context", "goals"]

    def add_example(self, example: pd.DataFrame) -> None:
        """Add new example to store for a key."""
        self.examples.append(example)

    def select_examples(self, num_examples: int = 5) -> list[dict]:
        """Select which examples to use based on the inputs."""
        num_examples = min(num_examples, len(self.examples))
        selected_rows = self.examples.sample(n=num_examples, replace=False, random_state=42)
        
        selected_examples = []
        for _, row in selected_rows.iterrows():
            example = {}
            for key in self.keys:
                example[key] = row[key]
            selected_examples.append(example)

        return selected_examples
    

def transform_fct(inputs: dict) -> list:
    """
    Transform raw output of description sampler into a list of dictionaries,
    each of which can be used as input to the semantic parser.
    """

    print("Executing transform fct")
    print(inputs["goals"])
    if "\n" in inputs["goals"]:
        utterances = [
            re.sub("^-|(\d)+", "", u).strip()  
            for u 
            in inputs["goals"].split("\n") 
            if len(u) > 3
        ]
    else:
        utterances = [
            re.sub("^-|(\d)+", "", u).strip().replace("\n", "") 
            for u 
            in inputs["goals"].split(".") 
            if len(u) > 3
        ]
    print("Cleaned utterances ", utterances)
    return {"goals_list": utterances}

def transform_utts_goals_fct(inputs):
    print("Transform goals and utts ", inputs)

    utterances = inputs["utterances"].split("\n")
    utterances = [u for u in utterances if len(u) > 3]
    print("utterances only ", utterances)
    goals_utterances_list = []
    for goal in inputs["goals_list"]:
        for u in utterances:
            goals_utterances_list.append({
                "goal": goal,
                "sentence": u,
            })   
    print("goals_utterances_list: ", goals_utterances_list)
    return {"goals_utterances_dict": goals_utterances_list}

def sample_knowledge(model, temperature, num_facts=1, **kwargs):
    """
    Function for retrieving knowledge statements
    for generated knowledge prompting.
    """
    # read in instructions
    instructions_text = f"""Instructions:
    You will be given a context in which a person asks a question. 
    Your task is to generate {str(num_facts)} possible goals the person might have in mind when asking the question.
    Provide the goals in a bullet point list.

    Examples: 
    """

    example_template = """
    Context: {context}
    Goals: {goals}    
    """

    # read in examples
    examples = pd.read_csv("data/session5/qa_examples.csv", sep = "|")
    
    example_selector = RandomExampleSelector(examples)
    selected_examples = example_selector.select_examples(num_examples=2)

    model = init_model(
        model_name=model, 
        temperature=temperature, 
        logprobs=0,
    )

    example_prompt = PromptTemplate(
        template = example_template,
        input_variables = ['context', 'goals'],
    )
    input_template = """
    Context: {context}
    Goal:
    """
    # format the few_shot prompt
    few_shot_prompt = FewShotPromptTemplate(
        prefix=instructions_text, 
        examples=selected_examples,
        example_prompt=example_prompt, 
        input_variables=["context"],
        suffix=input_template,
        example_separator="\n\n",
    )    

    goals_chain = LLMChain(
        llm=model, 
        prompt=few_shot_prompt, 
        verbose=True,
        output_key="goals"
    )

    # parse the outputs
    transform_chain = TransformChain(
        input_variables=["goals"], 
        output_variables=["goals_list"], 
        transform=transform_fct, 
        verbose=True
    )
    
    # define chain for sampling utterances
    instructions_utterances = f"""Instructions:
    You will be given a context about several objects. Imagine another person asks you what you have. Generate three well formed sentences telling another person about one of the available objects.
    """
    utterance_template = instructions_utterances + """
    Context: {context_utterance}
    Sentences: 
    """
    utterance_prompt = PromptTemplate(
        template = utterance_template,
        input_variables = ['context_utterance'],
    )
    utterance_chain = LLMChain(
        llm=model, 
        prompt=utterance_prompt, 
        verbose=True,
        output_key="utterances"
    )
    # parse the outputs into utt X goals dict
    transform_utterances_goals_chain = TransformChain(
        input_variables=["goals_list", "utterances"], 
        output_variables=["goals_utterances_dict"], 
        transform=transform_utts_goals_fct, 
        verbose=True
    )
   

    return goals_chain, transform_chain, utterance_chain, transform_utterances_goals_chain

def relevance_evaluator(input_dicts, model, temperature, **kwargs):

    pprint(input_dicts)
     # define chain for retrieving relevance of each utterance for each goal
    instructions_relevance = """Instructions:
    You will be given a person's goal and a sentence they might hear. On a scale between 0 and 10, how helpful is the sentence for the person to achieve their goal?
    """
    model = init_model(
        model,
        temperature=temperature,
    )
    scores = []
    for d in input_dicts:
        pprint(d)
        relevance_template = instructions_relevance + """
        Goal: {goal}
        Sentence: {sentence}
        Score:
        """
        relevance_prompt = PromptTemplate(
            template = relevance_template,
            input_variables = ['goal', 'sentence'],
        )

        relevance_chain = LLMChain(
            llm=model, 
            prompt=relevance_prompt, 
            verbose=True,
            output_key="score"
        )
        s = relevance_chain(d)["score"]
        scores.append(s.replace("\n", "").strip())

    print("Scores: ", scores)
    return scores



if __name__ == "__main__":

    goals_chain, transform_chain, utterance_chain, transform_utterances_goals_chain = sample_knowledge(
        "text-davinci-003",
        0.1,
    )

    overall_chain = SequentialChain(
        chains=[
            goals_chain, 
            transform_chain, 
            utterance_chain, 
            transform_utterances_goals_chain],
        input_variables=["context", "context_utterance"],
        output_variables=["goals_utterances_dict"],
        verbose=True
    )
    r = overall_chain(inputs={"context": " A woman walks into a gym and asks: Do you offer yoga classes?" , 
                               "context_utterance": "You work at the reception desk of a gym. The gym currently offers a pilates class, zumba classes, and kickboxing."})
    print("------ FINAL OUTPUTS ------ ", r)
    scores = relevance_evaluator(r["goals_utterances_dict"], "text-davinci-003", 0.1)
    print("------ FINAL OUTPUTS ------ ", scores)
   