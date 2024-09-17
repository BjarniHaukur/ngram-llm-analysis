# Team Diary

## Week 36 (September 2 - September 8)
Project Manager: Bjarni

### Bjarni
- **Number of hours this week: 7**
- Writing the project plan
- Getting to know our group members
- Familiarizing myself with the paper in question (and other relevant papers)
- Contemplating what exactly it is we are supposed to be understanding

### Boti
- **Number of hours this week: 5**
- Reading the related materials
- Writing the project plan, assessment with Johan via e-mail
- Thinking about possible approaches

### Jonas
- **Number of hours this week: 6.5**
- Familiarizing with the group
- Reading https://www.arxiv.org/abs/2407.12034
- Most time was spent on composing a project plan.
- Did some research on N-grams and found this great resource for efficient N-grams which we could use for N-gram-statistics. https://dl.acm.org/doi/abs/10.1145/3302913: Handling Massive N-Gram Datasets Efficiently, https://github.com/jermp/tongrams.

### Felix
- **Number of hours this week: 5.5**
- My primary focus was on organizing the collaboration aspects of the project and aligning the individual skill sets of each group member. We developed a strategy outlining how we intended to work together as a team, along with the methods necessary for our approach.
- I conducted research for our project plan, focusing on how N-Gram statistics have been historically applied to AI. Using the Web of Science, I identified the most relevant sources and potential high-impact papers.
- I revised and converted bullet points into continuous sentences to enhance the overall readability and flow of the content.

### Group Meeting
- We held a meeting to discuss potential projects, reviewed all available topics, and reached a consensus on a common direction. Following this, we sent the necessary emails.
- During the meeting, we introduced ourselves, discussed our skills and initial thoughts, and briefly talked about the chosen project. Afterwards, we all read Nguyen’s research paper.
- We developed the project plan in two stages: first, by establishing a clear structure, and then by highlighting the key points. Since we exceeded the two-page limit, we shortened sections on motivation and background, but kept these ideas for the final report.

### Peer Group Meeting
- We got in contact with the peer group and decided to set up a shared Discord Server. The first meeting was scheduled for the 9th of September.

### Supervisor Meeting
- We had a one-hour meeting to discuss our approaches and explore potential improvements to the project. We also reviewed the timeline for completing our tasks.
- Johan explained his choice of the N-Gram topic, highlighting its relevance to the explainability of transformers. He provided an example from the Swedish language, demonstrating how a single syllable can change the entire context.
- He also mentioned the possibility of acquiring resources from KTH to support our project.


---

## Week 37 (September 9 - September 15)
Project Manager: Bjarni

### Bjarni
- **Number of hours this week: TBD**
- Repurpose code from a similar project I had
    - BPETokenizer, MemmapDataset, project structure, sampling functions (sampling with temp, top_k sampling, nucleous sampling)
- Creating a small, workable, split of the TinyStories dataset
- Fitting A BPETokenizer to that split
    - Kind off similar to N-Gram
    - Reveals some "worrying" patterns in the data
    - "Once upon of time, there was a " is more common than "tree"


### Boti
- **Number of hours this week: TBD**

### Jonas
- **Number of hours this week: TBD**

### Felix
- **Number of hours this week: TBD**
1. Literature Review on Transformer Models and N-gram Statistics
I conducted an in-depth literature review on transformer models, with a particular emphasis on their relationship with n-gram statistics. This involved analyzing recent research papers, articles, and case studies to understand how n-gram approaches are integrated within transformer architectures. The objective was to identify methodologies that could enhance our project's performance by incorporating these statistical techniques.
2. Attempting to Run an N-gram Statistics GitHub Library on MacBook
I attempted to implement an open-source GitHub library designed for n-gram statistics on my MacBook. During this process, I encountered issues related to processor incompatibility. After extensive troubleshooting, I consulted with **Jonas**, who confirmed that the library is not compatible with certain MacBook processors. This setback highlighted the need to either modify the library for compatibility or seek alternative solutions that are compatible with our hardware.
3. Studying Previous Work on Tokenizers by **Bjarni** and **Jonas**
I invested time in understanding the previous work done by Bjarni and Jonas on tokenizers. This involved reviewing their codebases, documentation, and any supplementary materials they provided in order to further enhance my knowledge regarding AI models and close potential knowledge gaps.

### Brainstorming for Peer Group Meeting
1. Clarifying Our Understanding of Transformers Using This Method
We discussed what exactly we aim to understand about transformer models through our current method. It was noted that our approach might be describing the language dataset more than providing insights into the transformer architecture itself. We need to ensure that our methodology effectively analyzes the inner workings of transformers, not just the characteristics of the dataset.
2. Reproducing Paper Findings Before Setting Divergent Goals
We agreed to reproduce the results of the referenced paper before establishing new objectives that could diverge between groups. This will provide a solid foundation and ensure all team members have a shared understanding of the baseline findings.
Understanding "Model Variance" in Nguyen's Paper
3. We discussed the meaning of "Model Variance" as presented in Nguyen's paper. We concluded that this term refers to the stochasticity and variability that occur when the same model is trained multiple times. This variance demonstrates that the findings are significant and consistently observable across different training runs, rather than being artifacts of a single instance.

### Peer Group Meeting Results

**1. Progress Alignment with the Other Group**

- **Current Status:**
  - Both our group and the other team are at a similar stage in the project.
  - This parity offers an opportunity for collaboration and mutual support.

**2. Project Structure Review**

- **Discussion:**
  - We reviewed and discussed our finalized project structure.
  - Ensured that the methodology aligns with our research objectives and timelines.

**3. Understanding N-gram Statistics in Transformer Models**

- **Objective:**
  - To figure out how n-gram statistics can be effectively applied within the context of transformer-based models.
- **Challenges:**
  - Integrating n-gram analysis with transformers is complex due to the models' architecture.

**4. Implementation Challenges with N-gram Methods**

- **Other Group's Experience:**
  - Implemented an n-gram method but faced significant memory bottlenecks.
- **Consensus:**
  - Both teams agreed that developing a custom n-gram implementation may not be efficient.
  - Decided to explore existing tools to save time and resources.

**5. N-gram Definition and Marginalization**

- **Other Group's Approach:**
  - Using marginalization where the operation "−" reduces an n-gram to an (n−1)-gram.
- **Discussion:**
  - We explored this definition and its potential applicability to our project.
  - Considered the mathematical implications and how it affects data representation.

**6. Model Variance Discussion**

- **Our Perspective:**
  - Model variance refers to the variability that occurs when training multiple models.
  - Emphasized the importance of training multiple models to achieve statistically significant results, as per Nguyen's work.
- **Other Group's Perspective:**
  - Focused on overfitting criteria as an indicator of model variance.
- **Agreement:**
  - Both groups concluded that training multiple models is necessary, despite the computational intensity.

**7. Computational Intensity and Feasibility**

- **Concern:**
  - Training multiple models is computationally demanding.
- **Action Item:**
  - Discuss with our supervisor the necessity and feasibility of this approach.
  - Explore potential solutions, such as using more efficient algorithms or accessing additional computational resources.

**8. Use of Existing Libraries**

- **Other Group's Tool:**
  - Utilizing "tokengrams," a library available on GitHub ([Link](https://github.com/EleutherAI/tokengrams)).
- **Our Consideration:**
  - Evaluating whether adopting this library could mitigate the challenges we've faced.
  - Acknowledged that custom implementation may not be the best use of time.

**9. Secondary Dataset Selection**

- **Other Group's Status:**
  - Considering the open-source Wikipedia dataset but haven't made a final decision.
- **Our Plan:**
  - Intend to choose a different secondary dataset to allow for comparative analysis.
  - Potential datasets include [List Possible Datasets].

**10. Transformer Models and Tokenization**

- **Other Group's Approach:**
  - Using the GPT-2 transformer model from Hugging Face.
- **Our Challenges:**
  - Faced issues with GPT-2 due to its large vocabulary size, leading to increased complexity.
- **Our Solution:**
  - Developed a custom tokenizer to reduce the model size and complexity.
  - Aiming to make the model more manageable with our available computational resources.

**11. Need for Model Variance Analysis**

- **Question:**
  - Do we need to analyze model variance to substantiate our findings and ensure statistical relevance?
- **Decision:**
  - Agreed to consult with our supervisor to determine the importance of this analysis in our project scope.

**Action Items:**

1. **Consult Supervisor:**
   - Schedule a meeting to discuss the necessity of training multiple models and analyzing model variance.
2. **Evaluate "tokengrams" Library:**
   - Assess the feasibility of integrating the "tokengrams" library into our project.
3. **Dataset Selection:**
   - Research and select a secondary dataset different from the other group's choice.
4. **Optimize Tokenization:**
   - Continue refining our tokenizer to improve model efficiency.
5. **Weekly Meetings:**
   - Maintain regular meetings to monitor progress and collaborate effectively.

### Supervisor Meeting
- 

---

## Week 38 (September 16 - September 22)
Project Manager: Boti

### Bjarni
- **Number of hours this week: TBD**

### Boti
- **Number of hours this week: TBD**

### Jonas
- **Number of hours this week: TBD**

### Felix
- **Number of hours this week: TBD**

### Peer Group Meeting
- 

### Supervisor Meeting
- 

---

## Week 39 (September 23 - September 29)
Project Manager: Felix

### Bjarni
- **Number of hours this week: TBD**

### Boti
- **Number of hours this week: TBD**

### Jonas
- **Number of hours this week: TBD**

### Felix
- **Number of hours this week: TBD**

### Peer Group Meeting
- 

### Supervisor Meeting
- 

---

## Week 40 (September 30 - October 6)
Project Manager: Jonas

### Bjarni
- **Number of hours this week: TBD**

### Boti
- **Number of hours this week: TBD**

### Jonas
- **Number of hours this week: TBD**

### Felix
- **Number of hours this week: TBD**

### Peer Group Meeting
- 

### Supervisor Meeting
- 

---

## Week 41 (October 7 - October 13)
Project Manager: Bjarni

### Bjarni
- **Number of hours this week: TBD**

### Boti
- **Number of hours this week: TBD**

### Jonas
- **Number of hours this week: TBD**

### Felix
- **Number of hours this week: TBD**

### Peer Group Meeting
- 

### Supervisor Meeting
- 

---

## Week 42 (October 14 - October 20)
Project Manager: Boti

### Bjarni
- **Number of hours this week: TBD**

### Boti
- **Number of hours this week: TBD**

### Jonas
- **Number of hours this week: TBD**

### Felix
- **Number of hours this week: TBD**

### Peer Group Meeting
- 

### Supervisor Meeting
- 

---

## Week 43 (October 21 - October 27)
Project Manager: Felix

### Bjarni
- **Number of hours this week: TBD**

### Boti
- **Number of hours this week: TBD**

### Jonas
- **Number of hours this week: TBD**

### Felix
- **Number of hours this week: TBD**

### Peer Group Meeting
- 

### Supervisor Meeting
- 

---

## Week 44 (October 28 - November 3)
Project Manager: Jonas

### Bjarni
- **Number of hours this week: TBD**

### Boti
- **Number of hours this week: TBD**

### Jonas
- **Number of hours this week: TBD**

### Felix
- **Number of hours this week: TBD**

### Peer Group Meeting
- 

### Supervisor Meeting
- 

---

## Week 45 (November 4 - November 10)
Project Manager: Bjarni

### Bjarni
- **Number of hours this week: TBD**

### Boti
- **Number of hours this week: TBD**

### Jonas
- **Number of hours this week: TBD**

### Felix
- **Number of hours this week: TBD**

### Peer Group Meeting
- 

### Supervisor Meeting
- 

---

## Week 46 (November 11 - November 17)
Project Manager: Boti

### Bjarni
- **Number of hours this week: TBD**

### Boti
- **Number of hours this week: TBD**

### Jonas
- **Number of hours this week: TBD**

### Felix
- **Number of hours this week: TBD**

### Peer Group Meeting
- 

### Supervisor Meeting
- 

---

## Week 47 (November 18 - November 24)
Project Manager: Felix

### Bjarni
- **Number of hours this week: TBD**

### Boti
- **Number of hours this week: TBD**

### Jonas
- **Number of hours this week: TBD**

### Felix
- **Number of hours this week: TBD**

### Peer Group Meeting
- 

### Supervisor Meeting
- 

---

## Week 48 (November 25 - December 1)
Project Manager: Jonas

### Bjarni
- **Number of hours this week: TBD**

### Boti
- **Number of hours this week: TBD**

### Jonas
- **Number of hours this week: TBD**

### Felix
- **Number of hours this week: TBD**

### Peer Group Meeting
- 

### Supervisor Meeting
- 

---

## Week 49 (December 2 - December 8)
Project Manager: Bjarni

### Bjarni
- **Number of hours this week: TBD**

### Boti
- **Number of hours this week: TBD**

### Jonas
- **Number of hours this week: TBD**

### Felix
- **Number of hours this week: TBD**

### Peer Group Meeting
- 

### Supervisor Meeting
- 

---

## Week 50 (December 9 - December 15)
Project Manager: Boti

### Bjarni
- **Number of hours this week: TBD**

### Boti
- **Number of hours this week: TBD**

### Jonas
- **Number of hours this week: TBD**

### Felix
- **Number of hours this week: TBD**

### Peer Group Meeting
- 

### Supervisor Meeting
- 

---

## Week 51 (December 16 - December 22)
Project Manager: Felix

### Bjarni
- **Number of hours this week: TBD**

### Boti
- **Number of hours this week: TBD**

### Jonas
- **Number of hours this week: TBD**

### Felix
- **Number of hours this week: TBD**

### Peer Group Meeting
- 

### Supervisor Meeting
- 

---

## Week 52 (December 23 - December 29)
Project Manager: Jonas

### Bjarni
- **Number of hours this week: TBD**

### Boti
- **Number of hours this week: TBD**

### Jonas
- **Number of hours this week: TBD**

### Felix
- **Number of hours this week: TBD**

### Peer Group Meeting
- 

### Supervisor Meeting
- 

---

## Week 1 (December 30 - January 5)
Project Manager: Bjarni

### Bjarni
- **Number of hours this week: TBD**

### Boti
- **Number of hours this week: TBD**

### Jonas
- **Number of hours this week: TBD**

### Felix
- **Number of hours this week: TBD**

### Peer Group Meeting
- 

### Supervisor Meeting
- 
