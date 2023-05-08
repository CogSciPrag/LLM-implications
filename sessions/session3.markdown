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

The field of reinforcement learning (which is often considered a type of machine learning, next to supervised an unsupervised learning) has a long history and initially focused on optimal control in the 1950s. After some years of stagnation, it became increasingly popular in the 1980s and gained public awareness after 2015 when DeepMind used RL to create the famous AlphaZero. Now, RLHF (a specific instantiation of an RL-based optimization procedure) is attrackting increasing attention with the release of GPT-3.5 and GPT-4, as their naturalness and capacities might partly be due to this technology.

This session focuses on the *core conceptual aspects of RLHF* and omits theoretical aspects of RL which are irrelevant to the pipelines currently employed for training LLMs. In case you are curious about RL more boradly, you can find some useful resources at the end of the page.
After introducing RLHF, the lecture contains a quick tour over currently popular LLMs and dives into practically relevant aspects, namely, creating effective prompts for the models. Therefore, the learning goals for this session are:

* being able to identify purpose and motivation behind fine-tuning LLMs with RLHF
* understanding basics of RLHF and the following building blocks:
  * fine-tuning
  * reward model
  * PPO
* becoming familiar with recent LLMs
* bing able to use sophisticated prompting to control LLM output


### Slides

<!-- <object data="slides/03-RLHF-LLM-architectures.pdf" type="application/pdf" width="100%" height="500px"> 
      <p>Unable to display PDF file. <a href="slides/03-RLHF-LLM-architectures.pdf">Download</a> instead.</p>
    </object> -->

### Code

TBD.

### Additional resources

TBD.