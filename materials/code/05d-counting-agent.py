import os
import openai
from dotenv import load_dotenv
import json
from pprint import pprint
from utils import init_model

import langchain
from langchain import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.agents import load_tools, Tool
from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType

# read api key from file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def tool_generator(tooldict, tool_template, tool_description, model):
    """
    Produce a tool that can be used by an acrion agent.

    Parameters
    ----------
    tooldict : dict
        A dictionary with the keys "name" and "content". 
        The value of "name" is the name of the tool,
        and the value of "content" is the content of the tool.
        The content of the tool is (part of)
        the instructions that the tool's LLM gets.
    tool_template : str
        The string appended to all the tool's content
        in the tool's LLM call.
    tool_description : str
        The string appended to all the tool's content
        in the tool's description that the agent sees.
    model : LLM
        The instantiated LLM that the tool uses.
    
    Returns:
    --------
    tool : Tool
        The instantiated tool that can be used by an action agent.
    """

    tool_content = tooldict['content']
    tool_name = tooldict['name']

    counting_prompt = PromptTemplate(
        template=tool_content+tool_template,
        input_variables=['words'],
    )

    # Tool's internal LLM chain
    counting_chain = LLMChain(
        llm=model,
        prompt=counting_prompt,
        verbose=True,
        output_key="response"
    )

    def f(*query):
        # When the agent calls the tool with a certain input query,
        # the tool will return the output of the chain
        output = counting_chain(query)
        return output['response']

    tool = Tool(
        name=tool_name,
        # The description of the tool that the agent sees
        description=tool_content+tool_description,
        func=f
    )

    return tool


def run_letter_counting_agent(model, temperature):
    """
    Wrapper instantiating an action agent for counting letters in a list of words.

    Parameters:
    -----------
    model : str
        The name of the LLM model to be used.
    temperature : float
        The temperature for sampling from the LLM.
    Returns:
    --------
    output: str
        The final answer of the agent.
    """
    # define Agent backbone
    llm = init_model(
        model,
        temperature=temperature,
    )
    # load utils for loading hand-crafted tests
    with open("data/session5/letter_counter_single_tools.json", 'r') as f:
        tests = json.load(f)
    # input tempalte for all tools
    tool_template = "Words: {words}"
    # general description of all tools
    tool_description = "The input is a list of words. The output is the reasoning process for determining the number of letters in all words starting with a vowel and the final answer."
    # construct tools
    # use a predefined math tool
    tools = load_tools(["llm-math"], llm=llm) + [
        # generate a tool for each prompt ourselves
        tool_generator(
            test,
            tool_template,
            tool_description,
            llm,
        )
        for test in tests
    ]
    print("Tools: ")
    pprint(tools)
    # initialize action agent operating based on tool contents only
    agent = initialize_agent(
        tools, 
        llm, 
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
        verbose=True
    )

    words = """Polina, Michael, eggplant, cheese, oyster, imagination, elucidation, induce"""
    agent_instructions = """Your task is to figure out whether the number of letters in all words starting with a vowel sum up to 42. 
    You have some tools at your disposal to run some tests. If the tests give different results, you have to adjudicate between them and explain your choice. 
    Think step by step and use all tools. At the end, leave a line the only says 'yes' if the words do have the number of letters in sum and 'no' otherwise."""
    
    # run agent
    output = agent(agent_instructions + f"Words: {words}")
    print("--- Final output ---- ", output)


if __name__ == "__main__":
    run_letter_counting_agent(
        "text-davinci-003",
        0.3,
    )