---
layout: page
menu: main
title: Formalia
permalink: /overview/
---

You can find the main information about the course here.

The course is a seminar worth 6 ECTS by default. Note that the sessions are split into a **12:00 s.t.--13:30** block, a lunch break **13:30--14:15**, and the second half of the session **14:15--15:45**.

## Schedule

| session    | date       | topic                                  |
|------------|------------|----------------------------------------|
| 1          | April 25th | intro & overview                       |
| 2          | May 2nd    | core LLMs                              |
| 3          | May 9th    | prepped LLMs                           |
| 4 (online) | May 16th   | implications for linguistics           |
| 5          | May 23rd   | implications for CogSci                |
|            | May 30th   | **holiday**                            |
| 6          | Jun 6th    | implications for society               |
| 7          | Jun 13th   | optional initial project consultation  |
| 8          | Jun 20th   | project launch                         |
| ...        | ...        | project work                           |
| 9          | Jul 18th   | **intermediate project presentations** |
|            | Sep 1st    | **submission deadline**                |

## Final projects

In order to get credits for this course, you need to complete a final project in groups of 3-5 students.

### Important dates

The following important dates should be kept in mind!

- DATE X: communicate your chosen project & names of your group members to us via email to Polina
- June 13th: [optionally] attend the consultation session and talk to us about your selected project and any open questions you have
- June 20th: [obligatory] project launch: present your selected project in a 5 minute presentation
- July 18th: [obligatory] intermediate project presentation
- September 1st: [obligatory] final project submission deadline

### Project guidelines

*Preliminary* guidelines for the respective projects are the following:

* projects to be completed in groups of 3-5
* preliminary project presentations on July 18th
* final project submission deadline on September 1st 23:59 ECT
* the projects should result in submissions containing the following materials which will be the basis for grading:
  * building: 
    * intermediate presentation 
    * submit repository (or some other format of code sharing)  
    * submit 1-2 page project report
  * testing:
    * intermediate presentation
    * submit repository (or some other format of sharing materials used for the tests)
    * submit 1-2 page project report
  * creating: 
    * intermediate presentation
    * submit contents roughly corresponding to one in-depth topic discussion / group member 
    * submit some visualizations or other presentation aids 

### Miscellaneous organizational matters

- Feel free to use the dedicated Moodle Forum space for trying to find fellow group members.
- Sign up for consultations during the four weeks between project launch and intermediate presentations on Moodle if you'd like to talk to us about your project progress.

### Project ideas

As discussed in the slides for session 1, we suggest final projects that can roughly be grouped into the types "Build", "Test" and "Create".

#### 1. Build a "generative agent"
- reimplement a "generative agent" based on [Park et al. 2023 "generative agents"](https://arxiv.org/abs/2304.03442)
- use LangChain to implement a memory, planning, retrieval, reflection and action selection
- try to have two or more agents interact with each other in dialogue, e.g., in bargaining or argument

#### 2. Testing statistical world knowledge and pragmatic ability in LLMs
- A recent paper by Rohde et al. ([2022](https://direct.mit.edu/opmi/article/doi/10.1162/opmi_a_00058/112494/This-Better-Be-Interesting-A-Speaker-s-Decision-to)) investigated human expectations of relevance of information via a simple behavioral experiment.
- The goal of this project is to investigate whether LLMs have can be coerced to reproduced the human data, i.e., to see whether there is a sense in which LLMs have internalized statistical world knowledge and expectations of what counts as an appropriate, relevant communicative act.
- The paper showed examples like this one to human participants (all material is included in the paper) and ask them to select one of two numbers as the missing piece in a test sentence:
  - *Context*: Liam is a man from the US. Liam lives down the street from Rebecca.
  - *Test sentence*: Rebecca thinks / announced to me that Liam has ... T-shirts.
  - *Task question:* Choose the number which is more likely: 21 | 29
- The paper found that human decision makers are sensitive to the context (thinks/ announced to me, etc.) in the sense that unsolicited communicative acts raise the expectations of newsworthiness, which in turn corresponds to statistical unexpectedness (a higher number than one would normally expect).
- **Goal:** Test common LLMs for their ability to reproduced human decisions for experiments 1-3 from Rohde et al.'s paper.
  - Start with zero-shot prompting.
  - Add appropriate instructions, possibly examples with Chain-of-Thought prompting.
- **Possible extension:** Use LangChain to create a "pragmatic reasoning agent" in which (what you think are) relevant Chain-of-Thought steps are individually carried out by different calls to the LLM.

#### 3. Automatic creation of experimental materials
- create new material similar to existing experimental or benchmark data
  1. for testing LLM performance 
     - use ToM paper(s) to create additional vignettes / items
     - test original vs new material on ToM task (with otherwise identical prompts)
  2. [maybe] Jennifer Hu et al.'s pragmatic ability testing data set
     - data is indeed available
  3. [maybe] argstrength: the corpus by [Carlile et al. (2018)](https://aclanthology.org/P18-1058.pdf) contains essays annotated with respect to argument components, argument persuasiveness scores, and attributes of argument components that impact an argument’s persuasiveness. 
     - this might be interesting as materials for arg strength related projects
     - it is also interesting to investigate sensitivity of LLMs to argumentative strategies from the perspective of influencing the informational landscape.
     - the task might be to take the annotated passages of essays and try to vary them with respect to the quality of the single components of the argument. E.g., a passage might have an annotations with respect to persuasiveness (1 out of 4), eloquence (4 out of 6), relevance (5 out of 6). One could try to prompt LLMs in natural language to increase / decrese quality along single dimensions.
     - data can be found [here](https://www.hlt.utdallas.edu/~zixuan/EssayScoring/) (might time out first)
  4. [maybe] Linguistic benchmark based chain-of-thought annotations (CoT in a loose sense)
     - it would be interesting to have annotations of planning / reasoning steps accompanying solving "NLI"/"NLU" tasks from linguistic benchmarks, in a fashion similar to STREET
     - this would be helpful for fine-tuning various systems in the future, and interesting in order to understand what kinds of reasoning may be tested under the umbrella term "NLU"
     - example annotations to be produced for Swag (original task: select best sentence continuation out of 4) (tapping into 'world knowledge' about likely human actions, likely reasoning, likely conversations)
       -  On stage, a woman takes a seat at the piano. She
            a) sits on a bench as her sister plays with the doll.
            b) smiles with someone as the music plays.
            c) is in the crowd, watching the dancers.
            d) nervously sets her fingers on the keys.
       - CoT to be produced: Since the woman is on a stage and took a seat at the piano, it is likely that she is an artist and will perform something. She might be preparing for the performance and might open the piano or put out her music sheet next. Among the given sentences, sentence d) describes the most likely action of a person preparing to play the piano. 
  
#### 4. LLMs in education

* test current LLMs' performance on tasks important for effective and safe employment in educational contexts, as described by Bommasani et al. (2021, p. 67ff.). These tasks include providing helpful feedback and instructions to students (in various subjects).
* The goal of this project would be to test the applicability of current LLMs for educational purposes. A natural subject for this investigation is English or Maths. The proposal focuses on testing LLMs for purposes of L2 English learning.
* More specifically, the goal is to test whether:
  1. LLM captures learner's mistakes (accuracy)
  2. LLM provides correct feedback, i.e., explains what is wrong and why(quality)
  3. LLM provides valuable explanation of different phenomena.
* For instance, one could test their ability to do grammatical error correction for L2 English learners on the [WI+LOCNESS](https://www.cl.cam.ac.uk/research/nl/bea2019st/) corpus which contains corrected & annotated essays (download at Q&I+LOCNESS v2.1 -- not where user is prompted to submit a form). 
  * the corpus contains essays with various grammatical mistakes, like word order, determiner use, mistakes in tense and conditionals, and many more. 
* the task would be, e.g., to compare error correction performance and error explanations provided by LLMs to ground truth (maybe across prompting).
  * error explanations could be gathered from different sources, including online grammar learning resources.
* @PT can provide more corpora / background on grammatical error correction systems if necessary
* alternative corpora on Maths can be found in evaluations of current LLMs which are frequently tested on mathematical task solving. One good option might be STREET, which includes stepwise solutions.

#### 5. Create a "frame problem" data set
- In class, we discussed the "frame problem" as a foundational problem for classical AI. The example of the FP from class was that of a bom attached to a cart in a hut. Rolling the cart out of the hut should (normally) imply that the bomb has also been removed from the hut. This is problematic for classical AI since all of the "inertia constraints" and their exceptions (things that do or do not change) need to be spelled out explicitly. But we also saw that at least some instances of LLMs give language output that suggests that some models do not suffer from the FP.
- As far as we are aware of, the question of whether LLMs do or do not "solve" the frame problem has not been addressed. Some natural language understanding data sets (like GLUE and derivatives) contain examples that are related to the frame problem, but this is not systematically explored (AFAWCT).
- To explore whether LLMs "solve" the frame problem, we could create a data set of stories / cases like the robot and the bomb in the hut, with dedicated questions and forced choice answers ("The bomb is in the hut" vs "The bomb is in front of the hut."). We can then systematically test the performance of different LLMs on this new data set.

#### 6. ... projects on outreach / philo / ethics ...
- summarize the state of the art of an important discussion around LLMs by creating a:
  - term paper
  - video summary
  - podcast
- topics can be chosen based on interest, for instance:
  - philosophical / ethical debates:
    - should AI source code be openly available?
    - what to prioritize: X-risks or Y-risks?
  - instructional video on how (not) to use LLMs in class / school

