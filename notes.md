author: Martin Tomasovic

sources: https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard#/?params=-1%2C9&official=true , https://medium.com/data-science-at-microsoft/evaluating-llm-systems-metrics-challenges-and-best-practices-664ac25be7e5

language models: 
    Masked: BERT, RoBERTa
    Autoregressive: GPT-2 (there are even small 137M, 380M),  Llama 2 (7B and 13B), TinyLlama 1.1B, Falcon ( 7B, 40B and 180B), MPT (6.7B), Mistral (7B),  Qwen2.5 (3B, 7B, 14B) (variants for coding, math exists)

Huggingface open-llm-leaderboard up to 7B only official providers best: 
1. ([tiiuae/Falcon3-7B-Instruct](https://huggingface.co/tiiuae/Falcon3-7B-Instruct))
2. ([Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct))
3. ([nvidia/AceInstruct-7B](https://huggingface.co/nvidia/AceInstruct-7B))
4. [internlm/internlm2_5-7b-chat](https://huggingface.co/internlm/internlm2_5-7b-chat)
5. [Qwen/Qwen2.5-7B-Instruct-1M](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-1M)

Evaluation
- for sentence completion (can evaluate gpt2 non instruct): [HellaSwag](https://rowanzellers.com/hellaswag/)
- other: 
    - [GLUE Benchmark](https://gluebenchmark.com/): GLUE (General Language Understanding Evaluation) benchmark provides a standardized set of diverse NLP tasks to evaluate the effectiveness of different language models
    - [SuperGLUE Benchmark](https://super.gluebenchmark.com/): Compares more challenging and diverse tasks with GLUE, with comprehensive human baselines
    - [TruthfulQA](https://github.com/sylinrl/TruthfulQA): Measures truthfulness of model responses
    - [MMLU](https://github.com/hendrycks/test): MMLU (Massive Multitask Language Understanding) evaluates how well the LLM can multitask

Perplexity - lower is better.
Perplexity can be interpreted as the geometric average branching factor of the language according to the model. It is a function of the language and the model. When considered
a function of the model, it measures how good the model is. The better the model, the lower the perplexity. When considered a function of the language, perplexity estimates the entropy, or complexity, of that language. In the end, the quality of a language model must be measured by its effect on the specific application for which it was designed. 

Training datasets: 
Common Crawl - from web pages, BooksCorpus - from books for long text, English Wikipedia, News Crawl - from news articles, [BUT-FIT/BUT-LCC](https://huggingface.co/datasets/BUT-FIT/BUT-LCC) - czech language

Merging models: [MergeKit](https://huggingface.co/blog/mlabonne/merge-models) - methods:
- Spherical Linear Interpolation (SLERP)
- [TIES](https://arxiv.org/abs/2306.01708) (I prefer this)
- [DARE](https://arxiv.org/abs/2311.03099)
- Passthrough
Example of merged model: [suayptalha/Falcon3-Jessi-v0.4-7B-Slerp](https://huggingface.co/suayptalha/Falcon3-Jessi-v0.4-7B-Slerp)