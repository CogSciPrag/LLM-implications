---
layout: page
title: Session 6
permalink: /materials/session6
---

## Lecture 

### Introduction

Over the last years, large language models, and *foundation models* (i.e., generic large pretained neural models that are fine-tuned for downstream tasks) in general, have been deployed in end-user facing products. For instance, the LLM that powers ChatGPT is now being used by millions of users across the world, and our internet use can hardly be thought of without AI powered Google translate. Due to such large scale deployment, neural models, and AI more broadly, have profound impacts of society. These range from concerns about differential quality of AI powered products in certain use cases, over regulation of such products to management of advanced AI-related risks.

In this session, we will touch upon these different topics related to societal impacts of LLMs and AI. This topic is subject to an even more heated and more public debate than any of the topics we talked about so far. For many of the aspects, there is not even a remotely definitive answer as to how to act upon impacts of AI. Furthermore, the state of the debates is rapidly changing, and the potential actions are an ongoing 'tug of war' between AI companies, policymakers, research institutions and other actors. 

Therefore, it is difficult to define learning goals for this session. Rather, it should be thought of as a speedride across several topics so as to become aware of discussions that shape the field of AI for the broad public. We warmly encourage you to deep dive into topics you find important and, most importantly, to be maximally critical and try to view the (future) debates against the backdrop of your own experience & knowledge (possibly contributed to by this class). 

### Slides

<object data="slides/06-society-ethics.pdf" type="application/pdf" width="100%" height="500px"> 
    <p>Unable to display PDF file. <a href="slides/06-society-ethics.pdf">Download</a> instead.</p>
</object>

### Code 

For completing final projects which include testing LLMs, we recommend using free opensource models hosted on the Huggingface Inference API and available via requests (i.e., you don't have to download the model weights and run the model on your machine).

In particular, we suggest using Flan-T5 and the OpenAssistant models.

A minimal script showing how to generate model responses from the API can be found [here](code/06-huggingface-api-access.py).

(Note that you need to get a Huggingface API token. For that, create a free account on Huggingface, log in to your profile, navigate to user account settings and generate a new token. Save it in your project env file.)

### Further Materials

- [Main source for many parts of the lecture: Bommasani et al., 2021. On the Opportunities and Risks of Foundation Models](https://arxiv.org/abs/2108.07258)
- [A webbook on Alignment](https://aisafetyfundamentals.com/ai-alignment-curriculum)

