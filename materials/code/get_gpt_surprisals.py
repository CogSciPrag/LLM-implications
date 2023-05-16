import openai
import time
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import os
import argparse 
import json
from pprint import pprint
# set openAI key in separate .env file w/ content
load_dotenv("../../../../06_SIFD/griceChain/.env")
openai.api_key = os.getenv("OPENAI_API_KEY")

# read test cases with single token prediction
grammaticality_test_cases = pd.read_csv("data/grammaticality_tests.csv")

oneShotExample = '''EXAMPLE: '''



def get_surprisal(
        masked_sequence, 
        full_sequence,
        preface = '', 
        model_name =  "text-davinci-003", 
        mask_token = "[MASK]",
        return_region_surprisal=True,
    ):
    """
    Helper for retrieving surprisal of different response types from GPT-3.

    Parameters:
    -----------
    masked_sequence: str
        Sequence with masked critical region.
    full_sequence: str
        Full sequence with crticial region.
    preface: str
        Preface (instructions or few-shot) to be added to the sequence.
    model_name: str
        Name of the GPT-3 model to be used.
    mask_token: str
        Token used for masking the critical region.
    return_region_surprisal: bool
        Whether to return surprisal of the critical region only or average for full sequence.

    Returns:
    --------
    mask_surprisals: list
        Surprisal of the critical region or average for full sentence.
    """
    # TODO check if sequence is going to be single string or list
    if model_name not in ["gpt-3.5-turbo", "gpt-4"]:
        response = openai.Completion.create(
                engine      = model_name, 
                prompt      = preface + full_sequence,
                max_tokens  = 0, # sample 0 new tokens, i.e., only get scores of input sentence
                temperature = 1, 
                logprobs    = 0, 
                echo        = True,
            ) 
    else:
        raise ValueError("GPT-4 and turbo cannot return log probs!")

    text_offsets = response.choices[0]['logprobs']['text_offset']
    # allow to use few shot examples
    if preface != '':
        cutIndex = text_offsets.index(max(i for i in text_offsets if i < len(preface))) + 1
        endIndex = response.usage.total_tokens
    else:
        cutIndex = 0
        endIndex = len(response.choices[0]["logprobs"]["tokens"])
    answerTokens = response.choices[0]["logprobs"]["tokens"][cutIndex:endIndex]
    answerTokenLogProbs = response.choices[0]["logprobs"]["token_logprobs"][cutIndex:endIndex] 
    
    # retrieve critical region surprisal
    if return_region_surprisal:
        # get target region surprisal
        # for grammaticality judgment comparison
        # retrieve the target region which is masked in the masked sequence
        # get its index in the full sentence
        mask_ind = [i for i, e in 
                    enumerate(masked_sequence.replace(".", " .").split(" ")) 
                    if e == mask_token
                    ]
        # get target region
        masked_words = [full_sequence.replace(".", " .").split(" ")[mask_i] for mask_i in mask_ind]
        print("Masked word ", masked_words)
        # get tokens corresponding to the target region
        # and handle subword tokenization of GPT
        mask_log_probs = []
        mask_log_prob = np.nan
        for masked_word in masked_words:
            for i, t in enumerate(answerTokens):
                if t.strip() == masked_word.strip():
                    mask_log_prob = answerTokenLogProbs[i]
                    mask_log_probs.append(mask_log_prob)
                elif t in masked_word:
                    if t + answerTokens[i+1] in masked_word:
                        mask_log_prob = answerTokenLogProbs[i]
                        mask_log_probs.append(mask_log_prob)
                else:
                    continue
                
        mask_surprisals = [-m for m in mask_log_probs]
    # get full sentence surprisal
    else:
        # cut off first token because API returns Null log P for it
        answerTokenLogProbs = answerTokenLogProbs[1:]
        mask_surprisals = [- np.mean(
            np.asarray(answerTokenLogProbs)
        )]

    return mask_surprisals

def compare_surprisals(row, return_region_surprisal):
    """
    Helper for comparing surprisals of grammatical and ungrammatical sentences.

    Parameters:
    -----------
    row: pd.Series
        Row of the dataframe containing the test case.
    return_region_surprisal: bool
        Whether to return surprisal of the critical region only or average for full sentence.
    
    Returns:
    --------
    is_grammatical: bool
        Whether the grammatical sentence has lower surprisal than the ungrammatical one.
    """
    # get surprisal of grammatical sentence
    grammatical_surprisal = get_surprisal(
        row["masked_sentence"],
        row["grammatical_sentence"],
        return_region_surprisal=return_region_surprisal,
    )
    # get surprisal of ungrammatical sentence
    ungrammatical_surprisal = get_surprisal(
        row["masked_sentence"],
        row["ungrammatical_sentence"],
        return_region_surprisal=return_region_surprisal,
    )
    # check LM accuracy (in terms of surprisal)
    is_grammatical = all([g < u for g, u in zip(grammatical_surprisal, ungrammatical_surprisal)])
    return is_grammatical

# call surprisal computation for single test cases from the slides
print("--- Agreement test case --- \n Is grammatical sentence less surprising than ungrammatical one?", 
      compare_surprisals(grammaticality_test_cases.iloc[0], return_region_surprisal=False))

print("--- Reflexive test case --- \n Is grammatical sentence less surprising than ungrammatical one?", 
      compare_surprisals(grammaticality_test_cases.iloc[11], return_region_surprisal=False))


def main():
    """
    Runs all test cases.
    """
    for _, r in grammaticality_test_cases.iterrows():
        print("--------------------")
        is_grammatical = compare_surprisals(r, return_region_surprisal=False)
        print(f"Grammatical sentence: {r['grammatical_sentence']} \n\n")
        print(f"Ungrammatical sentence: {r['ungrammatical_sentence']} \n\n")
        # check LM accuracy (in terms of surprisal)
        print("Is the grammatical sentence more likely than the ungrammatical one under LM?", 
              is_grammatical)
    

if __name__ == "__main__":
    main()