---
layout: page
title: Session 5
permalink: /materials/session5
---

<div align="center">
    <img src="../sessions/assets/session5.png" width="70%">
</div>

### Introduction

Based on the discussions of vast capabilities of LLMs on various tasks in the previous classes, it might be tempting to say that LLMs *understand* the tasks and, more broadly, the world. But what does it really mean to **understand** something? Building on the question whether LLMs might be human-like models of language production, processing or learning, what is an **explanatory model**, and can LLMs count as such? These are far from trivial questions that have been discussed in fields like philosophy, cognitive science and philosophy of science for many years. 

Therefore, in this session, we set out to discuss the implications of LLMs on (philosophy of) cognitive science, with a particular focus on understanding and explanations. First, key theoretical approaches to these concepts will be highlighted, and then we will see how different aspects of LLMs' understanding have been inspected in recent studies. Then, we will switch the point of view and look at LLMs from the perspective of usefully engineered tools which arguably contain rich contents mirroring many aspects of human language use. We will dive into the recently sky-rocketed framework *LangChain* and get ideas about how to make the most of LLMs' capabilities in creative ways, and how these could be embedded in scientific explanatory models.

Therefore, learning goals for this session are:
* understand understanding
  * of language
  * of the world
  * of models
* being able to identify important aspects of an explanatory theory in science
* get acquainted with hybrid explanatory models
* dive into LLM chains, agents & LangChain (hands on) 

The high-level goals of this session are to enable ourselves to think about explanatory and usefully engineered models and how these could complement each other, as well as critically evaluate what the relation between LLMs and understanding might be. 
The materials for the hands-on part are available in the Code section below. These are meant to be a potential boilerplate / inspiration for class projects.

### Slides

<object data="slides/05-Philo-CogSci-Models.pdf" type="application/pdf" width="100%" height="500px"> 
    <p>Unable to display PDF file. <a href="slides/05-Philo-CogSci-Models.pdf">Download</a> instead.</p>
</object>

### Code

Below, the links to the single scripts used in the hands-on LangChain demo are listed.

* LangChain scripts
  * [Knowledge-based QA (approximate) reimplementation (of Liu et al., 2022)](code/05a-knowledge-based-qa.py)
    * showcases usage of prompts, templates, transformations, retrieval of generation information from models and predefined computations based on LLM output results
  * [Explanatory model of QA](code/05b-qa-model.py)
    * showcases usage of a Chain, generation of alternatives and evaluations for a cognitive model
  * [Image captioner chain](code/05c-image-captioner.py)
    * showcases extension of LLM based chains to other I/O, storage
  * [Letter counting agent](code/05d-counting-agent.py)
    * showcases an action agent using different tools, including custom written tools
  * [Utils](code/utils.py)
    * necessary for running scripts above
* Data for few-shot prompting & instructions
  * [Few-shot prompting examples for fact generation for knowledge-based QA](code/data/knowledge_examples.csv)
  * [Few-shot prompting examples for alternatives generation for the explanatory model](code/data/qa_examples.csv)
  * [List of tools consisting of original prompts from session 3 for the agent](code/data/letter_counter_tools.json)
  * [List of tools consisting of single step tools for the agent](code/data/letter_counter_single_tools.json)
    * showcases multi-tool agent performance

### Further materials

* [Official LangChain tutorials](https://python.langchain.com/en/latest/additional_resources/youtube.html)
* [Medium blog post introduction to LangChain](https://towardsdatascience.com/getting-started-with-langchain-a-beginners-guide-to-building-llm-powered-applications-95fc8898732c)
* [Generative agents paper](https://arxiv.org/abs/2304.03442)
* [Tree-of-thought: a paper suggesting an alternative to CoT prompting for solving various decision making tasks](https://arxiv.org/abs/2305.10601)
