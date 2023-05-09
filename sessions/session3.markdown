---
layout: page
title: Session 3
permalink: /materials/session3
---

<div align="center">
    <img src="../sessions/assets/session3.png" width="60%">
</div>

### Introduction

In this session, we will take a look at the core technology used to convert LLMs pretrained on plain language modeling into maximally helpful, flexible and interactive assistants --- **R**einforcement **L**earning from **H**uman **F**eedback (RLHF). The core idea of this approach is to adapt the models so that they satisfy human preferences. More specifically, they are trained on *human feedback* which formalizes *goals* (e.g., being helpful, honest, harmless) by rewarding model outputs which satisfy these goals, and dispreferring those that don't. The computational framework that allows to implement such learning and actually train a neural network is *reinforcement learning* (RL). 

The field of reinforcement learning (which is often considered a type of machine learning, next to supervised an unsupervised learning) has a long history and initially focused on optimal control in the 1950s. After some years of stagnation, it became increasingly popular in the 1980s and gained public awareness after 2015 when DeepMind used RL to create the famous AlphaZero. Now, RLHF (a specific instantiation of a deep RL based optimization procedure) is attrackting increasing attention with the release of GPT-3.5 and GPT-4, as their naturalness and capacities might partly be due to this technology.

This session focuses on the *core conceptual aspects of RLHF* and omits theoretical aspects of RL which are irrelevant to the pipelines currently employed for training LLMs. In case you are curious about RL more boradly, you can find some useful resources [here](#rl-resources).
After introducing RLHF, the lecture contains a quick tour over currently popular LLMs and dives into practically relevant aspects, namely, creating effective prompts for the models. Therefore, the learning goals for this session are:

* being able to identify purpose and motivation behind fine-tuning LLMs with RLHF
* understanding basics of RLHF and the following building blocks:
  * fine-tuning
  * reward model
  * PPO
* becoming familiar with recent LLMs
* being able to use sophisticated prompting to control LLM output

Equipped with a conceptual understanding of how LLMs are built, we will start already touch upon important topics like *alignment*, *truthfulness* and *interpretability*.

### Slides

<!-- <object data="slides/03-RLHF-LLM-architectures.pdf" type="application/pdf" width="100%" height="500px"> 
      <p>Unable to display PDF file. <a href="slides/03-RLHF-LLM-architectures.pdf">Download</a> instead.</p>
    </object> -->

Links to materials mentioned in this session can be found [here](#bib).

### Code

[Python script](code/03-FLAN-T5-generations.py) from the lecture for GPT-2 and FLAN T5 generations.

<h3 id="rl-resources">Additional resources on RL</h3>

These resources are a short suggested selection among infinitely many (possibly better, except for the textbook) further resources out there:

* [intuitive video about RLHF](https://youtu.be/PBH2nImUM5c)
* [series of great videos about deep RL in general by one of the most well-known researchers in the field, Pieter Abbeel](https://www.youtube.com/playlist?list=PLwRJQ4m4UJjNymuBM9RdmB3Z9N5-0IlY0)
* [OpenAI blogpost about PPO](https://openai.com/research/openai-baselines-ppo)
* [an intuitive video about PPO, but it presupposes some general knowledge of RL](https://www.youtube.com/watch?v=5P7I-xPq8u8)
* [*the* textbook about RL (detailed and in-depth)](https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf)
* [high-level blogpost about the development of the field](https://towardsdatascience.com/reinforcement-learning-fda8ff535bb6)

<h3 id="bib">Materials used in the session</h3>

Below are links to papers which were referenced in the lecture but weren't linked to:

* [Sparks of AGI](https://arxiv.org/abs/2303.12712)
* [RL textbook](https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf)
* [InstructGPT](https://arxiv.org/abs/2203.02155)
* [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)
* [One of first RLHF fine-tuned LMs](https://arxiv.org/abs/2009.01325)
* [LLaMA](https://arxiv.org/abs/2302.13971)
* [PaLM](https://arxiv.org/abs/2204.02311)
* [Gopher](https://arxiv.org/abs/2112.11446)
* [Flan-T5](https://arxiv.org/pdf/2210.11416.pdf)
* [GPT-3](https://arxiv.org/abs/2005.14165)
* [GPT-4](https://arxiv.org/abs/2303.08774)
