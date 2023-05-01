---
layout: page
title: Session 2
permalink: /materials/session2
---

<div align="center">
    <img src="../sessions/assets/session2.png" width="50%">
</div>

### Introduction

In this session, we provide an overview of the fundamental concepts behind **core language models**. That is, we take a look at training language models to predict the next word, also called *pre-training* -- the step that is taken before pre-trained language models undergo further fine-tuning to become more helpful and harmless assistants (like, e.g., ChatGPT). By taking a tour over some technical details of language model mechanics and how they are evaluated in the literature, the main learning goals we are aiming for in this session are: 

* understand what the core *language modeling objective* is (conceptually)
* understand the concepts behind different language model architectures
* understand the approach to evaluating trained language models.

That is, this session is meant to provide an overview of concepts crucial for a better understanding of more complex state-of-the-art systems and being able to discuss them based on a more solid conceptual background, rather then equip you with all details necessary for building core LMs from scratch.  

### Slides

TBD.

### Hands-on exercises

Below, you can find code for trying out pretrained models via the [Huggingface library](https://huggingface.co/) yourself. Additionally, there are snippets for looking at the effect of a critical moving part of LMs -- the decoding scheme -- in action.

#### Running the code

In order to explore, you don't have to write a lot of code by yourself -- you can use many handy functions allowing you to download models, tokenize text (i.e., map natural language (sub)words to numbers which a machine can use for computations), sample predictions of the pretrained models. To try different decoding schemes, you will only have to take a look at the documentation and change some parameters of the functions. So even if you have never written code for using LLMs before we encourage you to take on the challenge and try running the code and doing the exercises below.

We strongly encourage you to use Colab (a Jupyter server hosted by Google) to create a Jupyter notebook and paste the code below into it. The only requirement fot that is a Google account. You can find instructions on how to use Colab notebooks [here](https://colab.research.google.com/).

If you are a more advanced programmer and really want to run the code locally on your machine, we strongly encourage you to create an environment (e.g., with Conda) before you install any dependencies, and please keep in mind that pretrained language model weights might take up quite a bit of space on your hard drive or might require high RAM for prediction.

#### Generating predictions from models

Below is the code for loading the pretrained GPT-2 model and generating some sentence completions with it.

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

# display results
print("Predicted text:\n\n ", output_text)
```

**Exercise**

Besides core language models which generate text continuations (i.e., simply predict next tokens), Huggingface offers a variety of models fine-tuned for different other tasks. These often come with different model *heads*. 

1. Create a few example inputs for GPT-2 in a different language than English. Does the quality of the prediction intuitively match the quality of English predictions?
2. Load a version of BERT fine-tuned for question answering by adapting the code above (the Huggingface model ID is 'deepset/bert-base-cased-squad2') and create or find three examples of contexts and questions which could be answered based on information in context. What does the task-specific head for this model do?

#### Comparing decoding schemes

One import aspect which determines the quality of the text produced by language models is the *decoding strategy*. The code below shows how to use several popular decoding schemes with GPT-2. Of course, they can also be applied to other generative language models. 
It is taken from [this](https://michael-franke.github.io/npNLG/06-LSTMs/06d-decoding-GPT2.html) chapter of Michael's webbook on natural language generation. 

**Exercise**

1. Compare the resulting output for the input above based on different decoding schemes. Do you observe differences? Why do you think this happens? What are possible disadvantages of the different decoding schemes? 

```python
# helper function for decoding the prediction and printing
def pretty_print(s):
    print("Output:\n" + 100 * '-')
    print(tokenizer.batch_decode(
        s, 
        skip_special_tokens=True)
    )

# use function 'model.generate' from `transformer` package to sample by
greedy_output = model.generate(
    text_encoded,     # context to continue
    max_length=50,    # return maximally 50 words (including the input given)
)

pretty_print(greedy_output)

# In a pure sampling approach, we just sample each next word with exactly the probability assigned to it by the LM.
sample_output = model.generate(
    text_encoded,        # context to continue
    do_sample=True,   # use sampling (not beam search (see below))
    max_length=50,    # return maximally 50 words (including the input given)
    top_k=0           # just sample one word
)

pretty_print(sample_output)

# In simplified terms, beam search is a parallel search procedure that keeps a number of path probabilities open at each choice point, dropping the least likely as we go along.
beam_output = model.generate(
    text_encoded, 
    max_length=50, 
    num_beams=5, 
    early_stopping=True
) 

pretty_print(beam_output)
```

#### Loading evaluations

In the lecture we discussed different metrics and benchmarks for evaluating language models. Below you can find the code for loading a Wikipedia-based test dataset frequently used to compute the test performance of models. 

**Exercise**

The code below provides the negative log likelihood of the first 50 tokens in the dataset under GPT-2. Based on the NLL, compute  the perplexity of this sequence under GPT-2.

```python
# import Huggingface package managing open source datasets
from datasets import load_dataset

test = load_dataset("wikitext", split="test")
encodings = tokenizer(
    "\n\n".join(test["text"]), 
    return_tensors="pt",
).input_ids

input_tokens = encodings[:,10:50]

pretty_print(input_tokens)

output = model(input_tokens, labels = input_tokens)
print("Average NLL for wikipedia chunk", output.loss.item())

### your code for computing the perplexity goes here ###
# perplexity = 
```

When evaluating generated text for tasks like translation or summarization, metrics like BLEU-n are employed. We will use the T5 model which was also trained to translate between English, German and French. 
The snippet below is based on [this](https://huggingface.co/docs/transformers/tasks/translation) tutorial.

**Exercise**
 
1. Use the code below to compute the BLEU score for the provided English sentence into German based on the prediction by the model and the provided gold standard.  
2. Try to paraphrase the gold standard translated sentence slightly and compute the BLEU score again. What happens to the score and why?
3. Find some additional examples of translations between different languages [here](https://huggingface.co/datasets/wmt14) or come up with some examples sentences yourself (e.g., a sentence in your native language and its English translation).

```python
# import the implementation of the bleu score computation
from torchtext.data.metrics import bleu_score
# load model and tokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

tokenizer_t5 = T5Tokenizer.from_pretrained("t5-small")
model_t5 = T5ForConditionalGeneration.from_pretrained("t5-small")

# define example sentences for translating from English to German
text_en = "All of the others were of a different opinion."
text_de = "Alle anderen waren anderer Meinung."
# define task 
prefix = "translate English to German: "

# define helper function taking prediction and gold standard
def compute_bleu(prediction, target, n):
    """
    Parameters:
    ----------
    prediction: str
        Predicted translation.
    target: str
        Gold standard translation.
    n: int
        n-gram over which to compute BLEU.
    Returns:
    -------
    score: float
        BLEU-n score.
    """
    score = bleu_score(
        [prediction.split()], 
        [ target.split()], 
        max_n=n, 
        weights = [1/n] * n,
    )
    return score 

# encode the source and the target sentences
encoding_en = tokenizer_t5(
    [prefix + text_en],
    return_tensors="pt",
).input_ids
# we don't need the task prefix before the target
encoding_de = tokenizer_t5(
    [text_de],
    return_tensors="pt",
).input_ids

# predict with model
predicted_de = model_t5(encoding_en)

# decode the prediction
predicted_decoded_de = tokenizer_t5.batch_decode(
    predicted_de,
    skip_special_tokens=True,
)

# compute BLEU-1 for the prediction
### YOUR CODE CALLING THE HELPER ABOVE GOES HERE ###
# bleu1 = 
```