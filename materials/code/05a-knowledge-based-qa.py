import os
import pandas as pd
from dotenv import load_dotenv
import re
import numpy as np
from langchain import PromptTemplate, FewShotPromptTemplate
from langchain.prompts.example_selector.base import BaseExampleSelector
from langchain.example_generator import generate_example
from langchain.chains import TransformChain
from langchain.chains.llm import LLMChain
import openai

from utils import init_model

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class RandomExampleSelector(BaseExampleSelector):
    """
    Convenience class for loading few-shot examples from a file.
    """
    
    def __init__(self, examples: pd.DataFrame, num_facts=1):
        self.examples = examples
        self.num_facts = num_facts
       
        self.keys = ["input", "knowledge"]

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
    Transform raw output of the facts sampler (string)
    into a list of strings for easier further processing

    Parameters:
    -----------
    inputs: dict
        Dict with the key "knowledge" containing the proposed facts.
    Returns:
    --------
    facts: dict
        Dict with they key "facts" with a list of facts.
    """

    if "\n" in inputs["knowledge"]:
        facts = [
            re.sub("^-|(\d)+", "", u).strip() 
            for u 
            in inputs["knowledge"].split("\n") 
            if len(u) > 3
        ]
    else:
        facts = [
            re.sub("^-|(\d)+", "", u).strip().replace("\n", "") 
            for u 
            in inputs["knowledge"].split(".") 
            if len(u) > 3
        ]
    return {"facts": facts}   


def sample_knowledge(model_name, temperature, num_facts=2, **kwargs):
    """
    Function for retrieving knowledge statements
    for generated knowledge prompting.
    Includes a string postrprocessing step.

    Parameters:
    -----------
    model_name: str
        Name of the model to be used.
    temperature: float
        Temperature for sampling.
    num_facts: int
        Number of facts to be sampled per input question.
    **kwargs:
        Args for the backbone LLM.

    Returns:
    --------
    knowledge_chain: langchain.chains.llm.LLMChain
        Chain for generating knowledge statements. Takes question as input.
    transform_facts_chain: langchain.chains.TransformChain
        Chain for parsing the raw output of the knowledge sampler.
    """
    # read in instructions
    instructions_text = f"""Instructions:
    Generate {str(num_facts)} numerical fact(s) about obejcts. Please provide the facts in a bullet list.
    
    Examples: 
    """
    # define template for few-shot examples
    example_template = """
    Input: {input}
    Knowledge: {knowledge}    
    """

    # read in examples
    examples = pd.read_csv("data/session5/knowledge_examples.csv", sep = "|")
    # sample random examples from file
    example_selector = RandomExampleSelector(examples)
    selected_examples = example_selector.select_examples(num_examples=2)
    # instantiate the LLM backbone
    if model_name == "text-davinci-003" or model_name == "gpt-4":
        model = init_model(
            model_name=model_name, 
            temperature=temperature, 
         )
    elif "flan-t5" in model_name:
        print("Initting HF model")
        model = init_model(
            model_name=model_name, 
        )
    else:
        raise ValueError(f"Model {model_name} cannot be used for knowledge based QA.")

    # parse few-shot examples into template
    example_prompt = PromptTemplate(
        template = example_template,
        input_variables = ['input', 'knowledge'],
    )
    input_template = """
    Input: {input}
    Knowledge:
    """
    # format the few_shot prompt
    few_shot_prompt = FewShotPromptTemplate(
        prefix=instructions_text, 
        examples=selected_examples,
        example_prompt=example_prompt, 
        input_variables=["input"],
        suffix=input_template,
        example_separator="\n\n",
    )    
    # define the LLM
    knowledge_chain = LLMChain(
        llm=model, 
        prompt=few_shot_prompt, 
        verbose=True,
        output_key="knowledge"
    )

    # parse the outputs into list
    transform_facts_chain = TransformChain(
        input_variables=["knowledge"], 
        output_variables=["facts"], 
        transform=transform_fct, 
        verbose=True
    )
    
    return knowledge_chain, transform_facts_chain

def answer_question(question, answers, knowledge, model_name, temperature, **kwargs):
    """
    Function for answering multiple choice questions
    based on question and knowledge statements (rough replication of system by Liu et al, 2022).

    Parameters:
    -----------
    question: str
        Question to be answered and for which facts are generated.
    answers: list
        List of possible answers.
    knowledge: list
        List of knowledge facts (all are used for answer scoring).
    model_name: str
        Name of the model to be used.
    temperature: float
        Temperature for sampling.
    Returns:
    --------
    answer: str
        Answer with highest probability when conditioned on the facts.
    """
    answer_logprobs = []
    # define the template for scoring answers based on facts
    template = "{knowledge} {question} {answer}"
    # instantiate the LLM backbone
    # in particular, add params for retrieving log probs
    if model_name == "text-davinci-003":
        model = init_model(
            model_name=model_name, 
            temperature=temperature, 
            logprobs=0,
            max_tokens=0,
            echo=True,
        )
    elif "flan-t5" in model_name or "gpt-4" in model_name:
        model = init_model(
            model_name="text-davinci-003", 
            temperature=temperature, 
            logprobs=0,
            max_tokens=0,
            echo=True,
        )
    else:
        raise ValueError(f"Model {model_name} cannot be used for knowledge based QA.")
    
    # format the prompt
    prompt = PromptTemplate(
        template=template,
        input_variables = ['question', 'answer', 'knowledge'],
    )
    qa_chain = LLMChain(   
        llm=model,
        prompt=prompt,
        verbose=True,
    )
    # plain model request in order to get answer probabilities
    for answer in answers:
        # note that all facts are used for scoring
        result = qa_chain.generate(input_list=[{
            "question": question,
            "answer": answer,
            "knowledge": knowledge,
            }]
        )
        # retrieve log probs from LLM results object       
        log_p = result.generations[0][0].generation_info["logprobs"]["token_logprobs"]
        # cut off none probability of first token
        answer_logprobs.append(np.sum(np.array(log_p[1:])))
    # renormalize
    answer_logprobs = np.array(answer_logprobs)/np.sum(np.array(answer_logprobs))
    # find max probability
    max_prob_idx = np.argmax(answer_logprobs)
    # return answer with max probability
    print("All answers ", answers)
    print("Answer probabilities ", answer_logprobs)
    print("Selected answer ", answers[max_prob_idx])
    return answers[max_prob_idx]


if __name__ == "__main__":

    question = "Q: How many legs does a dog have?. A:"
    answers = [
        "one",
        "two",
        "three",
        "four",
        "zero",
    ]
    model_name =  "google/flan-t5-xxl" #"gpt-4" # #"text-davinci-003" #
    facts_chain, parser_chain = sample_knowledge(
        model_name,
        0.1,
    )
    output = facts_chain.predict(input=question)
    output_parsed = parser_chain(output)

    print("--- Predicted knowledge statement ---- \n", output_parsed["facts"])

    # answer question
    answer = answer_question(
        question,
        answers,
        output_parsed["facts"],
        model_name,
        0.1,
    ) 