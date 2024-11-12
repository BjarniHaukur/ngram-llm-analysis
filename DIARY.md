# Team Diary

## Week 46 (11.11.2024 - 17.11.2024)
Project Manager: Boti

### Bjarni
- **Number of hours this week: TBD**

### Boti
- **Number of hours this week: TBD**

### Jonas
- **Number of hours this week: TBD**

### Felix
- **Number of hours this week: TBD**
- Meeting with Jonas and Bjarni regarding the draft report and experiments due until then

### Peer Group Meeting
### Summary of Opponent Group Meeting on N-gram LLM Project
1. **Experiment Status**
    - The opposing group will not complete any experiments by Friday, indicating potential challenges with their workflow or approach.
2. **Approach Differences and Complexity Handling**
   - **Rule Selection**: The opposing group averages 30 seconds for optimal rule selection in a single context, working with over 600 rules. This method is distinct from our approach, as we use a faster trie implementation, making direct comparisons challenging.
   - **Smoothing Approach**: Discussion with Johan highlighted that smoothing was omitted by both groups. Nguyen’s work only applies “stupid backoff,” possibly concealing the absence of smoothing, which raises questions about optimal model accuracy and transparency.
3. **Reimplementation and Dataset Comparisons**
   - **Nguyen’s Results on TinyStories**: We shared our reimplementation of Nguyen’s work with the other group, achieving identical results on the TinyStories dataset.
   - **Tiny Textbooks Dataset**: In contrast, applying n-gram prediction on the Tiny Textbooks dataset did not show the same downward trend in accuracy, leading to the hypothesis that such trends propagate only in simpler datasets.
   - **Next Steps - Wikipedia Dataset**: We plan to test the Wikipedia dataset, which was also used by Nguyen, to investigate if similar trends as with Tiny Textbooks emerge.
4. **Workflow and Group Comparison**
   - We have largely moved on to additional experiments and report writing, a progress milestone that the opposing group acknowledges with a degree of “envy.”
5. **Overfitting and Ground Truth Definitions**
   - **Overfitting**: Overfitting detection based on n-gram statistics is currently neglected, aligning with Nguyen’s approach, which does not employ n-gram statistics for this purpose.
   - **Ground Truth Definition**: Differing interpretations of “ground truth” emerged, underscoring a conceptual difference between the groups.
6. **Interest in Comparative Results**
   - Bjarni expressed curiosity regarding whether the opposing group will replicate similar findings, especially with the Tiny Textbooks dataset introduced during the meeting.

### Supervisor Meeting
- Scheduled for Thursday

---

## Week 45 (04.11.2024 - 10.11.2024)
Project Manager: Jonas

### Bjarni
- **Number of hours this week: TBD**

### Boti
- **Number of hours this week: TBD**

### Jonas
- **Number of hours this week: 6.5h**
- Worked on getting unsmoothed N-gram statistics. Implemented unsmoothed with Bjarni. Finally we're gettings some results! :D

### Felix
- **Number of hours this week: 12h**
1. **Documentation on N-gram Implementation**: Text was developed on the n-gram implementation, utilizing Boti’s notes and code to ensure thorough documentation of all key aspects of the implementation.
2. **In-depth Study and Analysis of Nguyen's Paper**: An extensive review of Nguyen's paper was conducted to distill and organize each procedural step within his methodology. This work aims to provide a detailed section in the report, outlining Nguyen's approach while highlighting divergences in the current implementation.
3. **Coordination and Discussion in Meetings**:Organized meetings facilitated discussions on testing results and Nguyen’s outcomes, with analysis centered on result alignment and exploring potential reasons for any discrepancies. These sessions also served as a collaborative space for reviewing project progress and refining strategic approaches.
4. **Discussion on the Impact of Smoothing**: The implications of excluding smoothing in the n-gram model were evaluated, focusing on how this choice affects model performance and interpretation. This discussion contributes valuable insights to the project’s implementation strategy. 

### Peer Group Meeting
- **Smoothing Issues**: Both groups are encountering significant difficulties with smoothing techniques for n-grams, especially given the complexity added by custom rulesets like marginalization.
- **Simplifying Assumption Consideration**: Due to these challenges, both groups are contemplating omitting smoothing as a simplifying assumption, acknowledging the potential impact on results.
- **Project Challenges**: Boti expressed concerns that the project might be reaching a point of diminishing returns, suggesting that further progress could be challenging.
- **Rust Implementation Review**: Bjarni has offered to review his Rust implementation in hopes of addressing some of these issues, though he noted his limited expertise in this area.

### Supervisor Meeting
- Scheduled for Thursday
---

## Week 44 (28.10.2024 - 03.11.2024)
Project Manager: Felix

### Bjarni
- **Number of hours this week: TBD**
- Met with Felix and Boti, did some work on the ngram statistics
- Debugging the ngram trie implementation
- Calculating the metrics for the ngram trie
- Do so in a separate process, so it doesn't block the training loop

### Boti
- **Number of hours this week: 8+**
- Some bugs were found in the ngram trie implementation, wrong cache initialization, division by zero in some cases
- Fixed the bugs
- Created subset functions for ruleset calculations, so we can calculate rules for suffix and subgram rulesets
- After the bugfixes the smoothing slowed down, but it is still manageable
- Wrote documentation for Python intellisense, so it is easier to work with
- Talked with Bjarni about the rule selection and metrics we want to log

### Jonas
- **Number of hours this week: 7h**
- worked on top1 accuracy. Integrating the ngram and transformer predictions. Tried to understand what is sought after in terms of the metrics. Thoughts regarding what sentences/contexts should be given to the calculations. 

### Felix
- **Number of hours this week: 8h**
- **Experiment Analysis (Nguyen’s Work)**  
  - Analyzed Nguyen's experimental setups, focusing on parameter choices (e.g., smoothing, sequence steps).
  - Documented insights for aligning our approach and identifying gaps
- **Clustered Parameters for Tracking**  
  - Grouped essential parameters (n-gram order, smoothing, dataset, memory).
  - Set up a system to track and compare parameters during training
- **Prepared Papers on New Experiment Ideas**  
  - Collected papers on advanced smoothing, adaptive n-grams, and OOV handling.
  - Identified potential new experiments for future model refinement
- **Organizational and Reporting Tasks**  
  - Drafted intro, problem statement, and methodology in report.
  - Updated task tracker, scheduled check-ins, and outlined sections per feedback.

### Peer Group Meeting
- The opponent group does not use smoothing, applying a specific rule (around 30 seconds for the rule).
- They encountered issues with GPU instances, noting the need for a larger instance to handle their requirements. One training run takes approximately 24 hours.
- The group is exploring the next steps, including conducting experiments with more complex datasets beyond TinyStories.
- They discussed dataset variance and its potential implications on the overfitting criterion.
- Clarification from Bjarni: The opponent group’s issue was related to GPU VRAM already being occupied on their instance, not a data loading problem.

### Supervisor Meeting
#### Server Usage
- **Server Utilization**: The team has successfully utilized the server without issues.

#### Experiment Focus and Dataset Selection
- **Infinigrams**: Testing infinigrams remains a goal if time permits, though priority is on trying a larger or improved dataset first.
- **Dataset Choice**: The team is considering using Wikipedia or GPT-2's dataset as the base for experiments.

#### N-gram Implementation and Library
- **New N-gram Library**: Boti published a library for n-gram statistics, overcoming a memory limitation by adopting a vector memory architecture from Facebook.
- **Smoothing Techniques**: After discussing various smoothing options, Boti’s Kneser approach was preferred.
- **Repository Review**: No existing n-gram repositories support marginalization and rule adaptation like Nguyen's work, prompting Johan’s suggestion to publish their solution.

#### Experiment Analysis and Metrics
- **Nguyen’s Experiments**: The team analyzed Nguyen's experiments, gathering the metrics necessary for replication.
- **N-gram vs. Transformer Comparison**: Bjarni anticipates the transformer will perform better, though the TinyStories dataset shows only a minor difference.

#### Smoothing Discussion
- **Importance of Smoothing**: Bjarni emphasized that without smoothing, results may be biased, as transformers may not be accurately explained by n-grams, especially when token sequences aren’t present in training data. Johan agreed on the necessity, and Jonas recommended experimenting with smoothing.

#### Collaboration and Sharing Insights
- **Guidance to Other Group**: The team shared training insights to support the other group’s progress.
- **Smoothing Challenges**: Given that smoothing requires substantial computational resources, Johan noted this could disadvantage the other group due to their slower approach.

#### Report and Documentation
- **N-gram Implementation Details**: Johan requested a thorough description of the n-gram implementation for the report.

#### Testing and Training Plans
- **Logger Test**: Johan suggested a small test run to finalize the logger code before full-scale training.
- **Training Timeline**: The team aims to start training next week, recognizing that while plans are essential, flexibility is equally important.

#### Additional Notes
- **New Dataset and Tokenizer**: Changing the dataset would necessitate training a new tokenizer but shouldn’t present major issues.
- **Training Bug**: A conflict arose when another user started a training run simultaneously, indicating the need for coordination in scheduling training runs.
 
---

## Week 43 (21.10.2024 - 27.10.2024)
Project Manager: Boti

### Bjarni
- **Number of hours this week: 0**
- Focused on master's thesis applications

### Boti
- **Number of hours this week: 40+**
- Iterated the ngram trie implementation in Rust
- Created pybindings, published the first version of the library (ngram-trie) to PyPI
- Moved to using a logger for metrics on the speed
- Implemented parallel computation, it was still slow for smoothing, investigated why later
- Reworked the smoothing initialization, for faster and parallel computation
- Tried caching with data copying, but it used too much memory
- Reworked the trie so it can be cached efficiently, and still be used for parallel computation relatively fast
- Changed reference counting library (RcLite forked for some more functionality) to use less overhead and be a little bit faster
- Fine tuned cache sizes, fixed some bugs that presented with caching
- Found the bug in smoothing, we have to calculate the rules in order for the caching to be the most efficient (also the caching function was improved a bit)
- Moved ruleset to another object, so it is easier to work with
- Wrote documentation for the interface and tested the library
- Huge code cleanup, and rigorosly testing the speeds at every iteration, debugging some performance issues

### Jonas
- **Number of hours this week: 4h**
- Initialized ngram integration

### Felix
- **Number of hours this week: 6h**
- Worked on the report: wrote the introduction (context and objectives), problem description (overall scope of the project), and related work (summary and evaluation of relevant studies).
- Revised Nguyen's cited literature for better alignment and accuracy.
- Reviewed previously collected literature for additional insights - I used my notes to write the report in a detailed and sophiticated way
- Searched for new publications to include the latest research

### Peer Group Meeting
- N/A (cancelled on request of peer group)

### Supervisor Meeting

---

## Week 42 (14.10.2024 - 20.10.2024)
Project Manager: XXX

### Bjarni
- **Number of hours this week: TBD**

### Boti
- **Number of hours this week: 0**
- Exam week

### Jonas
- **Number of hours this week: 0h**
- Exam week

### Felix
- **Number of hours this week: 0h**
- No work performed due to exam season

### Peer Group Meeting
- N/A (cancelled due to exam season)

### Supervisor Meeting
- N/A (cancelled due to exam season)
  
---

## Week 41 (07.10.2024 - 13.10.2024)
Project Manager: Bjarni

### Bjarni
- **Number of hours this week: 1**
- Midterm week, little to no work
- Wrote some comments on our midterm report

### Boti
- **Number of hours this week: 20**
- Iterated the ngram trie implementation in Rust
- Code cleanup
- Switched to u16 for keys
- Tried BTreeMap for less memory usage, found sorted_vec_map (forked it so we can use some more functionality) and it used even less memory and was still fast
- Benchmarking memory usage and speed
- Fine tuned some default variables
- A bug still persists where if we fit the trie, the memory allocation is larger by ~25%, but when we load this trie from disk it stays maybe the correct amount. Still investigating.

### Jonas
- **Number of hours this week: 2h**
- Midterm report improvement. Adding substance to the report by integrating Bjarni's valuable comments on Felix's draft.

### Felix
- **Number of hours this week: 8h**
- Conducted an **in-depth review of n-gram statistics**, analyzing key patterns and metrics for future experiments
- **Summarized meetings** and structured previous work, ensuring the project’s progression remained clear and organized, while synthesizing important discussions for future reference.
- **Authored the midterm report**, compiling critical findings and documenting the project’s current status, methodologies, and next steps, ensuring clarity and cohesion in communication.
- **Prepared for upcoming meetings** by setting agendas, organizing to-dos, and facilitating smooth collaboration among team members, ensuring efficient task execution and goal alignment.

### Peer Group Meeting

#### Training Performance & Implementation
- **Training Speed**: The other group’s training is significantly slower than our group’s.
- **Training Loop**: We are using a custom training loop, while the other group is struggling with training speed.
- **Library Usage**: Our group uses a different library for training, which provides performance metrics not available in the standard implementation, as we use a partition. Bjarni pointed out these differences in libraries.
- **Training Loss**: We haven’t completed a long training run yet. During a 10-minute run, the training loss decreased from 8-9 to 2-3.
#### N-Gram & Dataset Setup
- **N-Gram Statistics**: Our group will only begin full training once the n-gram statistics are ready. Boti is working on the n-gram setup, which should be completed by Thursday.
- **Dataset**: The entire TinyStories Dataset is being used for training by our group.
- **Context Window**: The other group is using the default GPT-2 context window.
#### Feasibility & Accuracy
- **Feasibility Calculation**: Bjarni calculated that a 2-hour training run is feasible given the available FLOPS, but we will triple-check to ensure there are no errors. The large time difference between the groups’ implementations is a concern.
- **Accuracy**: We haven’t evaluated the accuracy yet since we haven’t done full training. The other group achieved 85% accuracy after 25 hours of training.
#### Group Collaboration & Communication
- **Assistance to the Other Group**: Bjarni plans to help the other group after the meeting.
- **Group Chat**: We decided to be more active in the group chat.

### Supervisor Meeting

#### 1. **Comparison with the Other Group**
   - **Approach**: 
     - Vast differences in approach between our group and the other group.
     - Our group focuses on optimization for both the LLM and the n-gram, while the other group uses an SQL database.
     - We designed our program to rely on keeping everything in memory, which contrasts with the other group's database reliance.
   - **Bottleneck**:
     - RAM is the primary bottleneck (64GB is currently sufficient for 80% of the Tiny Stories dataset).
     - We suggested increasing the memory to 128GB since the entire dataset would require 70GB.
#### 2. **Training and Optimizations**
   - **Training Time**:
     - The other group takes 10 days for 5 epochs of training, while we complete it in 11 hours.
   - **GPU Utilization**:
     - We discussed the optimizations we implemented to ensure optimal GPU utilization.
     - We conducted two training runs: the first crashed, but the second was successful, leading us to suspect an issue with the KTH cluster.
   - **Tokenization**:
     - Bjarne explained the issues encountered while not using pre-tokenization.
   - **Iterative Training**:
     - We train and validate on 1/100 of the epoch iteratively, and this idea was discussed with Johan.
   - **Model Evolution**:
     - We demonstrated how the model evolved during training, using a log file to track how tokenization changed over time.
#### 3. **Demonstrations**
   - **Live Demo**:
     - We gave Johan a live demo of the transformer in a Jupyter Notebook, using different examples to show the model’s behavior.
   - **Utils Folder**:
     - We suggested that Johan review the utils folder, as it contains the optimizations we implemented.
#### 4. **N-gram Implementation**
   - **Completion**:
     - Boti finished the n-gram implementation and optimized the marginalization process, ensuring redundancy is minimized.
   - **Next Steps**:
     - Debug the n-gram tree and implement Kneser-Ney smoothing, which Johan confirmed as state-of-the-art.
     - Consider using two servers—one for n-gram operations and one for training. 
#### 5. **Technical Adjustments**
   - **Cluster Modifications**:
     - We discussed modifying the KTH cluster setup:
       - **Option A**: Everything remains in the same pod, but we double the available RAM.
       - **Option B**: Add a second pod in the same network, using a socket to connect the GPU cluster to the new n-gram pod.
     - If these options aren't feasible, we suggested using only 70% of the Tiny Stories dataset.
   - **Memory Request**:
     - Johan agreed to email the system administrator immediately after the meeting to request additional memory.
   - **Parameter Limitations**:
     - Bjarni raised the question of how far we can push the current parameter limit of 140 million, given the resources at our disposal.
#### 6. **Dataset Considerations**
   - **Tiny Stories**:
     - We discussed moving on to another dataset, as Tiny Stories does not generate the most interesting results.
       
---

## Week 40 (30.09.2024 - 06.10.2024)
Project Manager: Jonas

### Bjarni
- **Number of hours this week: 8**
- Updated the training script
  - Dropped HF Trainer since it was too constraining for custom metric logging, in particular asynchronous logging of metrics
  - Implemented a custom training loop from scratch
  - Use all optimization tricks like fused AdamW, bfloat16, torch.compile, powers of 2, etc.
- Optimized the Tokenizer creation function, now can handle arbitrarily large datasets
- Optimized the MemmapDataset creation function, now can handle arbitrarily large datasets
- The Memmap enables fast training, eliminating any and all possible bottlenecks of the tokenization / dataloading process
- Currently 1 epoch of training on TinyStories takes around 2:20h with 140M params on the KTH cluster

### Boti
- **Number of hours this week: 20**
- Iterated the ngram trie implementation in Rust
- Implemented a testing bed for speed and memory usage statistics
- Benchmarked the speed and memory usage of the ngram trie implementation for some million tokens
- Researched state of the art multithreading solutions for Rust and how we could utilize them
- The dataset could even fit in under 64GB of RAM
- Researched smoothing algorithms (Chen & Goodman, 1998)
- Implemented Kneser-Ney smoothing

### Jonas
- **Number of hours this week: 6h**
- Setting up a server using Google Cloud. Struggled for some time to open up the firewall. Managed to do so after a while. 

### Felix
- **Number of hours this week: 8h**

#### 1. **Google Cloud Architecture Preparation**:
   - Set up and configure cloud infrastructure to support the n-gram model deployment.
   - Coordinate resource usage within the team to stay within budget and timelines.
#### 2. **Literature Review**:
   - Investigate alternative smoothing techniques mentioned in the reference paper.
   - Identify any gaps or potential flaws in the paper’s methodology.
   - Update references and ensure proper citation of key materials for future use.
#### 3. **Documentation**:
   - Keep detailed records of project progress, including key decisions and challenges.
   - Ensure documentation is ready for the upcoming 15-minute half-time evaluation presentation.
   - Contribute to the presentation materials, focusing on n-gram statistics, implementation, and findings.

### Peer Group Meeting

#### Key Decisions and Findings:
- We confirmed that our focus remains on 7-gram statistics.
- Boti’s analysis revealed that RUST is three times more efficient for managing the n-gram tree.
- We discussed the implementation's speed: compressed 3-gram data is around 500MB, with exponential growth expected for 7-grams.
#### Comparison with the Opponent Group:
- The opponent group sticks to using a database and doesn't store the tree in memory, unlike our approach.
- Their SQL file is compressed to 2.3GB locally, while we chose uncompressed data in memory to avoid time overhead.
- A single n-gram lookup takes around a second for them, compared to our μs range lookup time.
#### Potential Optimizations and Challenges:
- We explored the idea of making n-gram generation and training asynchronous in different processes and plan to test this.
- Unfortunately, the person responsible for implementation wasn't present, so we couldn’t dive into the technical details.
#### Paper Discussion:
- Both groups remain unclear about the alternative smoothing methods mentioned in the paper.
- We speculated whether the paper might have omitted details on certain potential flaws.
#### Upcoming Tasks:
- Next week, we have a half-time evaluation that requires a 15-minute Zoom presentation. While results aren't ready yet, we will focus on presenting our approach.
- We will prepare the presentation for Thursday, and Felix will take the lead on it.
#### Training Considerations:
- The opponent group has encountered problems during training.
- For us, the n-gram needs to be fully completed before starting training, as we plan to collect statistics during the training process.

### Supervisor Meeting
- N/A (Canceled due slow progression)
---

## Week 39 (September 23 - September 29)
Project Manager: Felix

### Bjarni
- **Number of hours this week: 8**
- Realized that any opensource implementation that uses suffix arrays is not compatible with marginalization
  - Realized this while making a barebones implementation using suffix arrays and binary search
  - Suffix arrays are lexicographically ordered and rely on binary search 
  - This does not work when searching with wildcards since they aren't lexicographically ordered (can be done with exhaustive search)
- Instead, I started implementing a suffix tree (trie) from scratch in Rust
  - Basically converting the python code from our previous implementation to Rust
- We discussed how we could avoid the problem of storing the entire ngram in memory
  - Realized that we were thinking in problems, not solutions
  - Sure, 7-grams for the entire TinyStories dataset may be 100GB+ in memory but we can easily rent a machine with 128GB of RAM for a few GC credits
  - Then we can do async API calls from our GPU cluster to this NGram server, ensuring we dont bottleneck training or halt while waiting for NGram lookups

### Boti
- **Number of hours this week: 15**
- Itarated the rust NGramTrie implementation
- Researched best practices for efficient ngram storage in Rust
- Researched multithreading solutions to trie insertions Rust
- Tried small example solutions and started testing them
- We can save the entire NGramTrie on disk and it would only take some 10GBs of space which is cheap

### Jonas
- **Number of hours this week: 4**
  - still sick.. Prescribed antibiotics from doctor
  - Fixed training loop so that it is easier to run. Dependencies such as dataset, tokenizer had to be initialized with `dataset.py` and `tokenizer.py`. Now its simple to use described in [README](README.md). [Commit](https://github.com/BjarniHaukur/ngram-llm-analysis/commit/565726f8a64bfb1f143299bdfdbabf6181d16bc9)

### Felix
- **Number of hours this week: 10h**
- Explored Google Cloud resources, focusing on NVIDIA A100 GPU and cost-saving strategies, including the free trial and $50 team coupons.
- Reviewed the ngram model and database solution, ensuring alignment with project goals and fit with Google Cloud
- Prepared project reports and documentation, summarizing key findings and strategies for the team’s next steps.

### Peer Group Meeting

#### 1. **Ngram Implementation: Competitive Status**
   - The competing group has discontinued their efforts to implement the ngram independently.
   - We conducted a detailed review of open-source ngram repositories on GitHub. None of them are compatible with marginalization because wildcards are not lexicographically ordered. As a result, they fail to meet our specific needs.

#### 2. **Workaround: Database Implementation**
   - To overcome the limitations of the available tools, our team has chosen to implement a database-based solution.
   - **Database Configuration**:
     - The database will store n-grams using a key attribute (token 1, token 2, ..., token n).
     - This structure allows for fast search queries but results in slower insertion speeds.

#### 3. **N-gram Size Debate**
   - A key point of discussion was whether to use 7-gram or 8-gram statistics for our model.
   - The competing group has already implemented 8-grams, storing 7 tokens.
   - We are still investigating the advantages and drawbacks of both options, with further research required before making a final decision.

#### 4. **Compute Resources: Strategic Cloud Usage**
   - The competing group is utilizing Johan’s cluster, but our team only has access to a partition of the KTH cluster.
   - Google Cloud, while offering better hardware, is costly, which poses a challenge for sustained use.

   **Team Resource Plan**:
   - Our team consists of four members, each with a $50 Google Cloud coupon.
   - We plan to use the NVIDIA A100 (80GB) GPU, which costs approximately $5 per hour. This gives us 40 hours of total compute time.
   - To maximize our efficiency, we are considering setting up a free trial Google Cloud instance with at least 100GB of RAM to store all n-grams in memory. This instance will be linked to the GPU cluster, which will either run on another Google Cloud instance or on the KTH cluster.

#### 5. **Innovative Approaches and Competitive Edge**
   - The competing group is focusing on early stopping. However, we have decided to avoid this approach and concentrate on broader dataset comparisons and model depth.
   - Given our limited compute resources, we aim to push the boundaries of n-gram models.
   
   **New Exploration: Infinite Grams**:
   - Inspired by recent research, we are exploring infinite grams to see if this method can improve our model's performance.
   - Although infinite grams do not support marginalization, we will evaluate if the potential benefits outweigh this limitation.

#### 6. **Next Steps**
   - Continue internal discussions on the 7-gram vs. 8-gram approach and finalize our decision.
   - Further investigate research papers on infinite grams to assess whether they can offer a competitive advantage.

### Supervisor Meeting
   - Scheduled for Thursday 3pm
---

## Week 38 (September 16 - September 22)
Project Manager: Boti

### Bjarni
- **Number of hours this week: 8**
- Looked at some open source repositories for N-Gram statistics
- Most seem insufficient for our needs, needing to be modified to support the different rulesets we want
- Before doing that I wanted to understand N-Grams better
    - Implemented an N-Gram trie in pure python to better understand everything.
    - Everything done in memory, no memmap etc.
    - Kneser-Ney smoothing implemented to better represent rare/unseen ngrams
    - Will probably not work for large datasets as is atleast
    - One idea I had is to "shard/chunk" the trie and only load parts of it into memory at a time when needded.

### Boti
- **Number of hours this week: 6**
1. Trying out the KTH cluster
2. Reading and understanding other solutions for n-grams, brainstorming for new solutions
3. Writing code and testing the size of the n-gram trie in memory and on disk

### Jonas
- **Number of hours this week: 4**
  - sick
  - Ngrams visualization. Visualize the ngrams generated. Failed to visualize the rules

### Felix
- **Number of hours this week: 6**
- **Transformer Research:** Studied key papers on transformer architectures using Web of Science to enhance our theoretical foundation, focusing on n-gram statistics.
- **Practical Insights:** Watched the lecture "**Let's Build ChatGPT from Scratch**" to gain applicable insights into transformer implementation.
- **Documentation and Coordination:** Organized reports, documented progress, and coordinated group activities to improve collaboration.

### Peer Group Meeting
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

### Supervisor Meeting
 - We discussed potential ways to simplify our model and incorporate approximations to reduce the computational load, ensuring that the ngram implementation remains efficient despite resource constraints.
    - It was decided that our group will hold regular meetings with the supervisor every **Thursday at 3 PM** to track progress, address challenges, and adjust strategies as needed.
    - We examined the recent release of GPT-01 and discussed its implications for natural language processing. The potential impact on ngram model development and comparisons to current architectures were considered.
    - Given the scarcity of computational resources, we agreed to reduce the complexity of the model. Instead of focusing on training a high-parameter model, we will prioritize a more manageable approach, which will better fit within our compute limits.

---

## Week 37 (September 9 - September 15)
Project Manager: Bjarni

### Bjarni
- **Number of hours this week: 10**
- Repurpose code from a similar project I had
    - BPETokenizer, MemmapDataset, project structure, sampling functions (sampling with temp, top_k sampling, nucleous sampling)
- Creating a small, workable, split of the TinyStories dataset
- Fitting A BPETokenizer to that split
    - Kind off similar to N-Gram
    - Reveals some "worrying" patterns in the data
    - "Once upon of time, there was a " is more common than "tree"
- Rewrote some code to support HuggingFace transformers and tokenizers and got training to run with logging to Weights and Biases


### Boti
- **Number of hours this week: 4**
1. Keeping up with the state and changes of the project
2. Reading the code

### Jonas
- **Number of hours this week: 4**
  - Sick
  - Checked out the N-grams repository [tongrams](https://github.com/JoRo-Code/tongrams)
      - realized that it was quite outdated and had incompatibility issues with M1 chips.

### Felix
- **Number of hours this week: 8**
1. Literature Review on Transformer Models and N-gram Statistics
I conducted an in-depth literature review on transformer models, with a particular emphasis on their relationship with n-gram statistics. This involved analyzing recent research papers, articles, and case studies to understand how n-gram approaches are integrated within transformer architectures. The objective was to identify methodologies that could enhance our project's performance by incorporating these statistical techniques.
2. Attempting to Run an N-gram Statistics GitHub Library on MacBook
I attempted to implement an open-source GitHub library designed for n-gram statistics on my MacBook. During this process, I encountered issues related to processor incompatibility. After extensive troubleshooting, I consulted with **Jonas**, who confirmed that the library is not compatible with certain MacBook processors. This setback highlighted the need to either modify the library for compatibility or seek alternative solutions that are compatible with our hardware.
3. Studying Previous Work on Tokenizers by **Bjarni** and **Jonas**
I invested time in understanding the previous work done by Bjarni and Jonas on tokenizers. This involved reviewing their codebases, documentation, and any supplementary materials they provided in order to further enhance my knowledge regarding AI models and close potential knowledge gaps.

### Peer Group Meeting
1. Clarifying Our Understanding of Transformers Using This Method
We discussed what exactly we aim to understand about transformer models through our current method. It was noted that our approach might be describing the language dataset more than providing insights into the transformer architecture itself. We need to ensure that our methodology effectively analyzes the inner workings of transformers, not just the characteristics of the dataset.
2. Reproducing Paper Findings Before Setting Divergent Goals
We agreed to reproduce the results of the referenced paper before establishing new objectives that could diverge between groups. This will provide a solid foundation and ensure all team members have a shared understanding of the baseline findings.
Understanding "Model Variance" in Nguyen's Paper
3. We discussed the meaning of "Model Variance" as presented in Nguyen's paper. We concluded that this term refers to the stochasticity and variability that occur when the same model is trained multiple times. This variance demonstrates that the findings are significant and consistently observable across different training runs, rather than being artifacts of a single instance.

     
### Supervisor Meeting
N/A

---

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

## Week X ( - )
Project Manager: NAME

### Bjarni
- **Number of hours this week: TBD**

### Boti
- **Number of hours this week: TBD**

### Jonas
- **Number of hours this week: TBD**

### Felix
- **Number of hours this week: TBD**

### Peer Group Meeting

### Supervisor Meeting

---
