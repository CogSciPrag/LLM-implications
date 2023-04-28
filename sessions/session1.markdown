---
layout: page
title: Session 1
permalink: /materials/session1
---

### Introduction

In this introductory session we discuss some recent examples to start the discussion of LLMs.

First, to get some impressions of capabilities of state-of-the-art LLMs, let's take a look at the video from the live demo of the  GPT-4 release, the latest closed-source model developed by OpenAI.

* [coding a Discord Python bot](https://www.youtube.com/live/outcGtbnMuQ?feature=share&t=382)
* ['TaxGPT'](https://www.youtube.com/live/outcGtbnMuQ?feature=share&t=1144)
* [additional example] [explaining an image](https://www.youtube.com/live/outcGtbnMuQ?feature=share&t=953)
* [additional example] [coding a website from napkin drawing](https://www.youtube.com/live/outcGtbnMuQ?feature=share&t=1048)

(Please note that examples for such demonstrations are likely to be prepared in advance and, therefore, might be cherry-picked. Therefore, it is difficult to say how the model will perform on other inputs -- we will have to see for ourselves.)

### Slides

<object data="slides/01-introduction.pdf" type="application/pdf" width="100%" height="500px">
      <p>Unable to display PDF file. <a href="slides/01-introduction.pdf">Download</a> instead.</p>
    </object>

### Where to try LLMs

Below you can find links where you could try cutting-edge LLMs yourself! Please note that accessing the fanciest models requires creating accounts. The use of some models by OpenAI might even require to pay. Using the OpenAI API for accessing the models also requires to pay once the free credits one gets upon signing up are used up. Please note that **you are not expected to pay** for using these models in this class. Please contact us if you do have any issues testing models that you would like to use.

* [ChatGPT](https://chat.openai.com/): This chat model provided by OpenAI requires an account, but allows you to play around for free using the nice graphical chat interface!
* [OpenAssistant](https://open-assistant.io/chat): This is an open source alternative to OpenAI's GPT based models, based on the foundation model LLaMA provided by Meta AI. Using it requires a free account and it also offers a nice graphical chat interface!
* [GPT-2](https://huggingface.co/gpt2-xl): This is the most advanced member of the GPT family that was open sourced by OpenAI. You can try it out via the graphical interface hosted as an inference playground by the framework Huggingface.
* [FLAN-T5](https://huggingface.co/google/flan-t5-xxl): Huggingface hosts a try-out playground for a large variety of other 'more classical' open-source models like BERT and T5. This is a link to trying out a fine-tuned version of T5, called FLAN-T5. It is also a transformer model (like GPT) which was fune-tuned on a variety of tasks.
* [advanced] [OpenAI API](https://platform.openai.com/docs/api-reference/introduction): If you are familiar with getting your hands dirty with LMs and would like to generate some text from the latest GPT models, feel free to have a look at the OpenAI API which allows you to generate predictions from the latest non open-source models. **WARNING:** you have to pay for using their API (either with the $20 free credit you receive when signing up or via your saved payment method), so please pay attention to how many tokens you generate and note that this is your own responsibility.
* [may be unavailable] [Alpaca](https://alpaca-ai.ngrok.io/): This is a fine-tuned version of the LLaMA foundation model provided by researchers from Stanford. Unfortunately, their graphical interface for trying the model tends to be unavailable. For general information about the model, check out [this](https://crfm.stanford.edu/2023/03/13/alpaca.html) page.
* [HuggingChat](https://huggingface.co/chat/): Free alternative to chatGPT based on Open Assistant's latest model.

There are several other big players (e.g., PALM by Google) in the field of LLMs which are not available for public testing or don't have graphical test interfaces. We will discuss currently main models in the next sessions in more detail.

### Additional materials to prepare for session 2 (optional)

The following is *additional* material, in case you would like to recap various topics (at different levels) which will come in handy in the next sessions. These are not a prerequisite for the next classes and we will walk through all the important concepts in class. Of course, these are just single suggestions in the sea of content on YouTube meant to kick start the recommendation algorithm -- you are welcome to browse and choose your own favorites :)

* [Basic Python recap (1h)](https://www.youtube.com/watch?v=kqtD5dpn9C8)
* [Python for machine learning & data science in general (50min)](https://www.youtube.com/watch?v=7eh4d6sabA0)
* [Introduction to neural networks](https://youtu.be/aircAruvnKk)
* [PyTorch for deep learning, a brief introduction (30 min)](https://youtu.be/IC0_FRiX-sw)
* [High level introduction to the transformers architecture in general](https://youtu.be/SZorAJ4I-sA)
* [Vaswani et al (2017): Attention is all you need. A walk through of one of the original papers on the transformers architecture](https://youtu.be/iDulhoQ2pro)
* [Introduction to the Huggingface library](https://www.youtube.com/watch?v=QEaBAZQCtwE)


[1]:{{ site.baseurl }}/slides/01-introduction.pdf