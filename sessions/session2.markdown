---
layout: page
title: Session 2
permalink: /materials/session2
---

### Introduction

In this session, we provide an overview of the fundamental concepts behind **core language models**. That is, we take a look at training language models to predict the next word, also called *pre-training* -- the step that is taken before pre-trained language models undergo further fine-tuning to become more helpful and harmless assistants (like, e.g., ChatGPT). By taking a tour over some technical details of language model mechanics and how they are evaluated in the literature, the main learning goals we are aiming for in this session are: 

* understand what the core *language modeling objective* is (conceptually)
* understand the concepts behind different language model architectures
* understand the approach to evaluating trained language models.

That is, this session is meant to provide an overview of concepts crucial for a better understanding of more complex state-of-the-art systems and being able to discuss them based on a more solid conceptual background, rather then equip you with all details necessary for building core LMs from scratch.  

### Slides

TBD.

### Hands-on exercises

Below, you can find code for trying out pretrained models via the Huggingface library yourself. Additionally, there are snippets for looking at the effect of a critical moving part of LMs -- the decoding scheme -- in action. 

#### Running the code

Colab / some prose.


#### Generating predictions from models

Below is the code for loading the pretrained GPT-2 model and generating some sentence completions with it.

Exercise: Load BERT and T5. Different languages. Different tasks. 

```python
# load packages
from transformers import AutoTokenizer, AutoModelForCausalLM

# load pretrained tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

# define input text
text = "Once upon a time a quick fox "

# tokenize input text
text_encoded = tokenizer(
    text,
    return_tensors="pt",
).input_ids

# generate continuation with greedy decoding
output = model.generate(
    text_encoded,
    max_new_tokens=128,
)

# decode output tokens from IDs to words
output_text = tokenizer.batch_decode(
    output,
    skip_special_tokens=True,
)

print("Predicted text:\n\n ", output_text)
```

#### Comparing decoding schemes

TBD.