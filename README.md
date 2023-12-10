# LLM-paper
本项目主要总结LLM相关论文和学习路线
## 论文
### Survey
* [A Survey of Large Language Models](https://arxiv.org/pdf/2303.18223.pdf)[[notes](notes/A_Survey_of_Large_Language_Models.md)]

### 基础模型
* [Transformers](https://arxiv.org/pdf/1706.03762.pdf)[[blog:illustrated-transformer](https://jalammar.github.io/illustrated-transformer/),[Arxiv Dive](https://blog.oxen.ai/arxiv-dives-attention-is-all-you-need/), [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)][[notes](notes/Transformer_notes.md)]
* [GPT-1](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)[[notes](notes/GPT-1.md)]
* [GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)[[Arxiv Dive](https://blog.oxen.ai/arxiv-dives-language-models-are-unsupervised-multitask-learners-gpt-2/)]
* [GPT-3](https://arxiv.org/pdf/2005.14165.pdf)

### Instruction Learning
* [instructGPT](https://arxiv.org/pdf/2203.02155.pdf)[[Arxiv Dive](https://blog.oxen.ai/training-language-models-to-follow-instructions-instructgpt/)]
* [self-instruct](https://arxiv.org/pdf/2212.10560.pdf)

### 轻量级训练技术
* [LoRA](https://arxiv.org/abs/2106.09685)

### 并行训练技术
* [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/pdf/1910.02054.pdf)
* [GPipe: Easy Scaling with Micro-Batch Pipeline Parallelism](https://arxiv.org/pdf/1811.06965.pdf)
* [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelis](https://arxiv.org/pdf/1909.08053.pdf)

### 量化技术
* [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/pdf/2211.10438.pdf)

### 强化学习
* [RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback](https://arxiv.org/abs/2309.00267)
* [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)

### Scaling Law
* [Training Compute Optimal Language Models](https://arxiv.org/abs/2203.15556)
* [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)

### LLM实例
* [Claude](https://arxiv.org/pdf/2212.08073.pdf)
* [LLaMA](https://arxiv.org/pdf/2302.13971.pdf)
* [LLaMA 2](https://arxiv.org/pdf/2307.09288.pdf)[[Arxiv Dive](https://blog.oxen.ai/arxiv-dives-how-llama-2-works/)]
* [Baichuan2](https://arxiv.org/pdf/2309.10305.pdf)
* [Sparks of Artificial General Intelligence: Early experiments with GPT-4](https://arxiv.org/abs/2303.12712)


### 磁盘/文件系统
* [Retrieval Augmented Generation (RAG)](https://arxiv.org/abs/2005.11401)[[Arxiv Dive](https://blog.oxen.ai/arxiv-dives-rag/)]
* [Demonstrate-Search-Predict: Composing retrieval and language models for knowledge-intensive NLP](https://arxiv.org/abs/2212.14024)

### 多模态
* [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
* [CLIP - Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
* [NExT-GPT: Any-to-any multimodal large language models](https://next-gpt.github.io/)
* [LLaVA - Visual Instruction Tuning](https://arxiv.org/abs/2304.08485)
* [LaVIN - Cheap and Quick: Efficient Vision-Language Instruction Tuning for Large Language Models](https://arxiv.org/abs/2305.15023)
* [Emu Video: Factorizing Text-to-Video Generation by Explicit Image Conditioning](https://arxiv.org/abs/2311.10709)
* [ULIP: Learning a Unified Representation of Language, Images, and Point Clouds for 3D Understanding](https://arxiv.org/abs/2212.05171)
* [CoCa: Contrastive Captioners are Image-Text Foundation Models](https://arxiv.org/abs/2205.01917)

### 工具
* [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761)
* [Large Language Models as Tool Makers](https://arxiv.org/abs/2305.17126)
* [ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs](https://arxiv.org/abs/2307.16789)

### 系统一&系统二思维
* Thinking Fast and Slow[book]
* [Mastering the game of Go with deep neural networks and tree search](https://www.nature.com/articles/nature16961)
* [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)
* [Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/abs/2305.10601)
* [System 2 Attention](https://arxiv.org/abs/2311.11829)

### 越狱LLM与安全
* [Jailbroken: How Does LLM Safety Training Fail?](https://arxiv.org/abs/2307.02483)
* [Universal and Transferable Adversarial Attacks on Aligned Language Models](https://arxiv.org/abs/2307.15043)
* [Visual Adversarial Examples Jailbreak Aligned Large Language Models](https://arxiv.org/abs/2306.13213)
* [Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection](https://arxiv.org/abs/2302.12173)
* [Hacking Google Bard - From Prompt Injection to Data Exfiltration](https://embracethered.com/blog/posts/2023/google-bard-data-exfiltration/)
* [Poisoning Language Models During Instruction Tuning](https://arxiv.org/abs/2305.00944)
* [Poisoning Web-Scale Training Datasets is Practical](https://arxiv.org/abs/2302.10149)

## 代码
* [Running an LLM locally](https://blog.oxen.ai/how-to-run-llama-2-on-cpu-after-fine-tuning-with-lora/)[[Llama.cpp code](https://github.com/ggerganov/llama.cpp),[Andrej’s code](https://github.com/karpathy/llama2.c/blob/master/run.c)]
* [gpt4all](https://github.com/nomic-ai/gpt4all)
 


参考：
[Reading List For Andrej Karpathy’s “Intro to Large Language Models” Video
](https://blog.oxen.ai/reading-list-for-andrej-karpathys-intro-to-large-language-models-video/)
