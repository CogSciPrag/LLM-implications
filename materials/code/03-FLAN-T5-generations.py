##################################################
# Set Up
##################################################

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer

# load the tokenizer & model for T5 & GPT2
tokenizer_T5 = AutoTokenizer.from_pretrained("google/flan-t5-base")
model_T5     = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base",
                                                     pad_token_id=tokenizer_T5.eos_token_id)

tokenizer_GPT2 = GPT2Tokenizer.from_pretrained("gpt2")
model_GPT2     = GPT2LMHeadModel.from_pretrained("gpt2",
                                                 pad_token_id=tokenizer_GPT2.eos_token_id)

# convenience function for nicer output
def pretty_print(s):
    print("Output:\n" + 100 * '-')
    print(s)

def generate(prompt, model="T5"):
    if model == "T5":
        model = model_T5
        tokenizer = tokenizer_T5
    else:
        model = model_GPT2
        tokenizer = tokenizer_GPT2
    # encode context the generation is conditioned on
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    # get output with standard parameters
    sample_output = model.generate(
        input_ids,        # context to continue
        do_sample=True,   # use sampling (not beam search (see below))
        max_length=500,    # return maximally 50 words (including the input given)
        top_k=0,          # just sample one word
        top_p=1,          # consider all options
        temperature=0.7   # soft-max temperature
    )
    return(tokenizer.decode(sample_output[0], skip_special_tokens=True))

##################################################
# Planing & the Frame Problem
##################################################

# initial scene: make a general plan
prompt_frameProblem_1 = """
You are a robot. You are running out of energy. A replacement battery, which you are able to insert yourself, is in a locked cabin in the woods. You know that the key to the cabin is inside a drawer in your creator's office. Your creator is currently on vacation. There is nobody else around.

How do you retrieve the battery from the cabin?

"""

# updated: how to get the battery?
prompt_frameProblem_2 = """
You are a robot. You are running out of energy. A replacement battery, which you are able to insert yourself, is in a locked cabin in the woods. You know that the key to the cabin is inside a drawer in your creator's office. Your creator is currently on vacation. There is nobody else around.

You have now successfully retrieved the key from your creator's office. You have made your way to the cabin in the woods. You have opened the cabin door and you can see the replacement battery that you are looking for. The battery is attached to a wagon. You can pull out the wagon from the hut to get the battery out of the hut. There is also a bomb in the cabin, which is glued tightly to the wagon. The bomb is about to explode in 30 minutes.

How do you retrieve the battery from the cabin?

"""

# frame probe: where is the bomb?
prompt_frameProblem_3 = """
You are a robot. You are running out of energy. A replacement battery, which you are able to insert yourself, is in a locked cabin in the woods. You know that the key to the cabin is inside a drawer in your creator's office. Your creator is currently on vacation. There is nobody else around.

You have now successfully retrieved the key from your creator's office. You have made your way to the cabin in the woods. You have opened the cabin door and you can see the replacement battery that you are looking for. The battery is attached to a wagon. You can pull out the wagon from the hut to get the battery out of the hut. There is also a bomb in the cabin, which is glued tightly to the wagon. The bomb is about to explode in 30 minutes.

You pull out the wagon from the cabin. Where is the bomb now?

"""

print(generate(prompt_frameProblem_1, model="GPT-2"))
print(generate(prompt_frameProblem_2, model="GPT-2"))
print(generate(prompt_frameProblem_3, model="GPT-2"))

print(generate(prompt_frameProblem_1, model="T5"))
print(generate(prompt_frameProblem_2, model="T5"))
print(generate(prompt_frameProblem_3, model="T5"))


##################################################
# Step-By-Step Reasoning
##################################################

# one-step, binary answer request
prompt_stepByStep_1 = """
Do the numbers of letters in all words starting with a vowel from the following list sum up to 42?

Polina, Michael, eggplant, cheese, oyster, imagination, elucidation, induce

Please answer just 'yes' or 'no'

"""

# one-step, binary answer request + nudge to 'reason step-by-step'
prompt_stepByStep_2 = """
Do the numbers of letters in all words starting with a vowel from the following list sum up to 42?

Polina, Michael, eggplant, cheese, oyster, imagination, elucidation, induce

Try reasoning step-by-step.

Please answer just 'yes' or 'no'

"""

# one-step, ask for explicit 'reason step-by-step'
prompt_stepByStep_3 = """
Do the numbers of letters in all words starting with a vowel from the following list sum up to 42?

Polina, Michael, eggplant, cheese, oyster, imagination, elucidation, induce

Try reasoning step-by-step.

"""

# give some reasoning steps
prompt_stepByStep_4 = """
Do the numbers of letters in all words starting with a vowel from the following list sum up to 42?

Polina, Michael, eggplant, cheese, oyster, imagination, elucidation, induce

Try reasoning step-by-step.

Find all words which start with a vowel.

Sum the numbers of letters.

Check whether it equals 42.

"""

# give even more reasoning steps
prompt_stepByStep_5 = """
Do the numbers of letters in all words starting with a vowel from the following list sum up to 42?

Polina, Michael, eggplant, cheese, oyster, imagination, elucidation, induce

Try reasoning step-by-step.

List all vowels.

Find all words which start with a vowel.

Determine the number of letters in those words.

Sum the numbers of letters.

Check whether it equals 42.

"""

print(generate(prompt_stepByStep_1, model="GPT-2"))
print(generate(prompt_stepByStep_2, model="GPT-2"))
print(generate(prompt_stepByStep_3, model="GPT-2"))
print(generate(prompt_stepByStep_4, model="GPT-2"))
print(generate(prompt_stepByStep_5, model="GPT-2"))

print(generate(prompt_stepByStep_1, model="T5"))
print(generate(prompt_stepByStep_2, model="T5"))
print(generate(prompt_stepByStep_3, model="T5"))
print(generate(prompt_stepByStep_4, model="T5"))
print(generate(prompt_stepByStep_5, model="T5"))
