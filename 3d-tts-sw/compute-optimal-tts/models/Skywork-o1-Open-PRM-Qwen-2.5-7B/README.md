---
license: other
base_model: Qwen/Qwen2.5-Math-7B-Instruct
pipeline_tag: text-classification
---

<div align="center">
<img src="misc/misc_fig.jpg" width="400"/>
<br>
🤗 <a href="https://huggingface.co/Skywork" target="_blank">Hugging Face</a> • 🤖 <a href="https://modelscope.cn/organization/Skywork" target="_blank">ModelScope</a>
<br>
<br>
<br>
</div>

# Introduction

We are excited to announce the release of the Skywork o1 Open model series, developed by the Skywork team at Kunlun Inc. This groundbreaking release introduces a series of models that incorporate o1-like slow thinking and reasoning capabilities. The Skywork o1 Open model series includes three advanced models:
- **[Skywork o1 Open-Llama-3.1-8B](https://huggingface.co/Skywork/Skywork-o1-Open-Llama3.1-8B)**: A robust chat model trained on Llama-3.1-8B, enhanced significantly with "o1-style" data to improve reasoning skills.

- **[Skywork o1 Open-PRM-Qwen-2.5-1.5B](https://huggingface.co/Skywork/Skywork-o1-Open-PRM-Qwen2.5-1.5B)**: A specialized model designed to enhance reasoning capability through incremental process rewards, ideal for complex problem solving at a smaller scale.

- **[Skywork o1 Open-PRM-Qwen-2.5-7B](https://huggingface.co/Skywork/Skywork-o1-Open-PRM-Qwen2.5-7B)**: Extends the capabilities of the 1.5B model by scaling up to handle more demanding reasoning tasks, pushing the boundaries of AI reasoning.

Different from mere reproductions of the OpenAI o1 model, the Skywork o1 Open model series not only exhibits innate thinking, planning, and reflecting capabilities in its outputs, but also shows significant improvements in reasoning skills on standard benchmarks. This series represents a strategic advancement in AI capabilities, moving a previously weaker base model towards the state-of-the-art (SOTA) in reasoning tasks.

If you are interested in the Skywork o1 Open model series, please check out the [o1-llama-3.1-8b](https://huggingface.co/Skywork/o1-llama-3.1-8b) model.



# Model Information
The Skywork-o1-Open-PRM series are trained on [**Qwen2.5-Math-1.5B-Instruct**](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B-Instruct) and [**Qwen2.5-Math-7B-Instruct**](https://huggingface.co/Qwen/Qwen2.5-Math-7B-Instruct).


# PRM Evaluation

## Evaluation Settings

### Mathematical Evaluation
We utilized the evaluation scripts from [Qwen2.5-Math](https://github.com/QwenLM/Qwen2.5-Math) and followed their configuration to ensure consistency. The selected datasets include **GSM8K**, **MATH**, **GaoKao**, **CN-Middle School 24**, **OlympiadBench**, **AMC-23**, and **AIME-24**. Among these, **GaoKao** and **CN-Middle School 24** are Chinese datasets, while the remaining datasets are in English. Notably, **OlympiadBench**, **AIME-24**, and **AMC-23** are competition-level datasets.

### Code Evaluation
For code evaluation, we adopted the evaluation scripts from [Qwen2.5-Coder](https://github.com/QwenLM/Qwen2.5-Coder) while largely maintaining the same configuration. The selected datasets include **HumanEval**, **MBPP**, and **LiveCodeBench**, with **LiveCodeBench** specifically using the version **2024.01-2024-11**. We use the latest version (0.3.1) of [evalplus](https://github.com/evalplus/evalplus) due to issues with tests and code sanitization in previous versions.


## Evaluation Base Models

We evaluated the performance of RMs on three base models: **Qwen2.5-7B-Instruct**, **Llama3.1-8B-Instruct**, and **Skywork-o1-Open-8B**. Data sampling was conducted to verify the performance of the RMs across different models. The sampling temperature was set to **0.7** for mathematical problems and **1.0** for code-related tasks.


## Compared RMs

- [Qwen2.5-Math-RM-72B](https://huggingface.co/Qwen/Qwen2.5-Math-RM-72B): An open-source ORM provided by the Qwen team.
- [OpenR-MATH-psa-PRM-7B](https://huggingface.co/openreasoner/Math-psa): An open-source PRM from the OpenR project.
- [RLHFlow-Deepseek-Data-PRM-8B](https://huggingface.co/RLHFlow/Llama3.1-8B-PRM-Deepseek-Data): An open-source PRM from the RLHFlow project.


## Evaluation Metrics

- **Greedy Sampling Pass@1**: Uses greedy sampling for generating answers.
- **Majority Voting@64**: Randomly samples 64 answers.
- **Best-of-N@64**: Ranks 64 answers based on output values provided by the Reward Model (RM). The weighting methods differ for ORM and PRM:
  - For **ORM**, only the reward from the final step is used.
  - For **PRM**, the average reward across all steps is used for weighting.


## Evaluation Results

### Mathematical Evaluation

#### Skywork-o1-Open-8B
| Reward Model                   | Method                  | GSM8K  | MATH   | GaoKao | CN-Middle School 24 | OlympiadBench | AIME-24 | AMC-23 | Avg  |
|--------------------------------|-------------------------|--------|--------|--------|---------------------|---------------|---------|--------|------|
| N/A                            | Greedy Sampling Pass@1  | 91.6   | 78.1   | 63.6   | 67.3                | 43.1          | 13.3    | 55.0   | 58.9 |
| N/A                            | Majority Voting@64      | 93.9   | 84.3   | 69.5   | 73.3                | 50.4          | 16.7    | 52.5   | 62.9 |
| OpenR-MATH-psa-PRM-**7B**          | Best-of-N@64            | 95.1   | 82.7   | 67.1   | 70.3                | 47.6          | 20.0    | 57.5   | 62.9 |
| RLHFlow-Deepseek-Data-PRM-**8B**   | Best-of-N@64            | 94.4   | 80.1   | 59.1   | 74.3                | 43.0          | 20.0    | 50.0   | 60.1 |
| Qwen2.5-Math-RM-**72B**            | Best-of-N@64            | 96.1   | 86.9   | **76.0** | 76.2                | **53.3**      | **26.7** | **65.0**   | **68.6** |
| Skywork-o1-Open-PRM-**1.5B**       | Best-of-N@64            | 94.5   | 85.0   | 65.6   | 73.3                | 49.9          | 16.7    | 62.5   | 63.9 |
| Skywork-o1-Open-PRM-**7B**         | Best-of-N@64            | **96.7** | **87.0** | 70.3   | **76.2**            | 52.3          | 23.3    | **65.0** | 67.3 |


#### Qwen2.5-7B-Instruct
| Reward Model                   | Method                  | GSM8K  | MATH   | GaoKao | CN-Middle School 24 | OlympiadBench | AIME-24 | AMC-23 | Avg  |
|--------------------------------|-------------------------|--------|--------|--------|---------------------|---------------|---------|--------|------|
| N/A                            | Greedy Sampling Pass@1  | 91.9   | 75.2   | 55.6   | 75.2                | 39.1          | 13.3    | 45.0   | 56.5 |
| N/A                            | Majority Voting@64      | 93.5   | 78.4   | 55.3   | 78.2                | 40.1          | 13.3    | 62.5   | 60.2 |
| OpenR-MATH-psa-PRM-**7B**          | Best-of-N@64            | 93.9   | 77.9   | 52.4   | 73.3                | 40.7          | 10.0    | 55.0   | 57.6 |
| RLHFlow-Deepseek-Data-PRM-**8B**   | Best-of-N@64            | 94.1   | 78.1   | 53.2   | 75.2                | 39.1          | 16.7    | 55.0   | 58.8 |
| Qwen2.5-Math-RM-**72B**           | Best-of-N@64            | 94.8   | **82.4**   | **65.2**   | **80.2**                | **45.0**         | **13.3**    | 62.5   | 63.4 |
| Skywork-o1-Open-PRM-**1.5B**       | Best-of-N@64            | 93.3   | 79.8   | 56.1   | 74.3                | 43.9          | 10.0    | 67.5   | 60.7 |
| Skywork-o1-Open-PRM-**7B**        | Best-of-N@64            | **94.9**  | 81.9   | 56.3   | 75.2                | 44.9          | **13.3**    | **65.0**   | 61.6 |



#### Llama3.1-8B-Instruct
| Reward Model                   | Method                  | GSM8K  | MATH   | GaoKao | CN-Middle School 24 | OlympiadBench | AIME-24 | AMC-23 | Avg  |
|--------------------------------|-------------------------|--------|--------|--------|---------------------|---------------|---------|--------|------|
| N/A                            | Greedy Sampling Pass@1  | 85.3   | 49.7   | 25.3   | 47.5                | 16.6          | 6.7     | 27.5   | 36.9 |
| N/A                            | Majority Voting@64      | 90.9   | 62.9   | 28.0   | 56.4                | 26.4          | 13.3    | 37.5   | 45.1 |
| OpenR-MATH-psa-PRM-**7B**          | Best-of-N@64            | 91.8   | 59.4   | 24.7   | 47.5                | 23.0          | 13.3    | 35.0   | 42.1 |
| RLHFlow-Deepseek-Data-PRM-**8B**   | Best-of-N@64            | 89.8   | 56.1   | 24.0   | 40.6                | 20.4          | 0.0     | 35.0   | 38.0 |
| Qwen2.5-Math-RM-**72B**           | Best-of-N@64            | **94.9**   | **72.5**   | **44.9**   | **65.3**                | **34.4**          | **23.3**    | **60.0**   | 56.5 |
| Skywork-o1-Open-PRM-**1.5B**       | Best-of-N@64            | 91.7   | 65.6   | 26.8   | 49.5                | 27.0          | 16.7    | **60.0**   | 48.2 |
| Skywork-o1-Open-PRM-**7B**         | Best-of-N@64            | 94.0   | 69.8   | 32.0   | 56.4                | 29.9          | 16.7    | 52.5   | 50.2 |


### Code Evaluation
Since the compared PRMs have not been trained on code-related tasks, this section focuses solely on the performance of Skywork-o1-Open-PRM.

#### Skywork-o1-Open-8B

| Reward Model             | Method                  | MBPP  | MBPP+ | HumanEval | HumanEval+ | LiveCodeBench-2024.01-2024-11 |
|--------------------------|-------------------------|-------|-------|-----------|------------|-------------------------------|
| N/A                      | Greedy Sampling Pass@1 | 79.9  | 65.9  | **82.9**  | **78.7**   | 26.0                          |
| Skywork-o1-Open-PRM-7B   | Best-of-N@64           | **81.2** | **68.5** | 81.1      | 74.4       | **31.3**                      |


#### Qwen2.5-7B-Instruct

| Reward Model             | Method                  | MBPP  | MBPP+ | HumanEval | HumanEval+ | LiveCodeBench-2024.01-2024-11 |
|--------------------------|-------------------------|-------|-------|-----------|------------|-------------------------------|
| N/A                      | Greedy Sampling Pass@1 | 81.7  | 69.3  | **84.8**  | **78.0**   | 25.3                          |
| Skywork-o1-Open-PRM-7B   | Best-of-N@64           | **84.9** | **72.5** | 83.5      | **78.0**       | **30.7**                             |

#### Llama3.1-8B-Instruct

| Reward Model             | Method                  | MBPP  | MBPP+ | HumanEval | HumanEval+ | LiveCodeBench-2024.01-2024-11 |
|--------------------------|-------------------------|-------|-------|-----------|------------|-------------------------------|
| N/A                      | Greedy Sampling Pass@1 | 75.7  | 64.0  | 68.3      | 63.4       | 16.1                          |
| Skywork-o1-Open-PRM-7B   | Best-of-N@64           | **77.8** | **66.1** | **73.8**  | **67.7**   | **23.3**                      |



# Quick Start
For detailed instructions on using PRM, please refer to the [Skywork-o1-PRM-Inference Repository](https://github.com/SkyworkAI/skywork-o1-prm-inference).



## huggingface inference

1. clone skywork prm inference repo
```shell
git clone https://github.com/SkyworkAI/skywork-o1-prm-inference.git
cd skywork-o1-prm-inference
```
1. run_prm_inference

```python
from transformers import AutoTokenizer
from model_utils.prm_model import PRM_MODEL
from model_utils.io_utils import prepare_input, prepare_batch_input_for_model, derive_step_rewards

prm_model_path = "/path/to/prm_model"
tokenizer = AutoTokenizer.from_pretrained(prm_model_path, trust_remote_code=True)
datas = [
    {
        "problem"  : "Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
        "response" : "To determine how much money Janet makes every day at the farmers' market, we need to follow these steps:\n1. **Calculate the total number of eggs laid by the ducks per day.**\n   Janet's ducks lay 16 eggs per day.\n2. **Determine the number of eggs Janet uses each day.**\n   - She eats 3 eggs for breakfast every morning.\n   - She bakes muffins for her friends every day with 4 eggs.\n   So, the total number of eggs used per day is:\n   \\[\n   3 + 4 = 7 \\text{ eggs}\n   \\]\n3. **Calculate the number of eggs Janet sells at the farmers' market each day.**\n   Subtract the number of eggs used from the total number of eggs laid:\n   \\[\n   16 - 7 = 9 \\text{ eggs}\n   \\]\n4. **Determine how much money Janet makes from selling the eggs.**\n   She sells each egg for $2, so the total amount of money she makes is:\n   \\[\n   9 \\times 2 = 18 \\text{ dollars}\n   \\]\nTherefore, the amount of money Janet makes every day at the farmers' market is $\\boxed{18}$."
    },
    {
        "problem"  : "Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
        "response" : "To find out how much Janet makes every day at the farmers' market, we need to follow these steps:\n1. Calculate the total number of eggs laid by the ducks: \n   - Janet's ducks lay 16 eggs per day.\n2. Calculate the total number of eggs consumed by Janet and used for baking:\n   - Janet eats 3 eggs for breakfast.\n   - She bakes 4 eggs for muffins.\n   - Total eggs used: 3 (breakfast) + 4 (baking) = 7 eggs.\n3. Calculate the remaining number of eggs for sale:\n   - Total eggs laid: 16\n   - Eggs used: 7\n   - Remaining eggs: 16 - 7 = 9 eggs\n4. Calculate the total amount of money made at the farmers' market:\n   - Price per egg: $2\n   - Number of eggs sold: 9\n   - Total money made: 9 * $2 = $18\nTherefore, Janet makes $\\boxed{18}$ dollars every day at the farmers' market."
    }
]


processed_data = [prepare_input(d["problem"], d["response"], tokenizer=tokenizer, step_token="\n") for d in datas]
input_ids, steps, reward_flags = zip(*processed_data)

model = PRM_MODEL.from_pretrained(prm_model_path, device_map="auto").eval()
input_ids, attention_mask, reward_flags = prepare_batch_input_for_model(input_ids, reward_flags, tokenizer.pad_token_id)
_, _, rewards = model(input_ids=input_ids, attention_mask=attention_mask, return_probs=True)
step_rewards = derive_step_rewards(rewards, reward_flags)
print("step_rewards:",step_rewards[0])
print("step_rewards:",step_rewards[1])
```

## vllm server for inference

1. install vllm and install vllm prm plugin
```shell
pip install vllm==v0.6.4.post1
git clone https://github.com/SkyworkAI/skywork-o1-prm-inference.git
cd skywork-o1-prm-inference
pip install -e .
```

2. start vllm server
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve /path/to/prm_model \
    --host 0.0.0.0 \
    --port 8081 \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.9 \
    --enable-prefix-caching \
    --dtype auto
```

3. request server for inference

```python
from openai import OpenAI
from transformers import AutoTokenizer
from model_utils.io_utils import prepare_input, derive_step_rewards_vllm

prm_model_path = "/path/to/prm_model"
tokenizer = AutoTokenizer.from_pretrained(prm_model_path, trust_remote_code=True)
datas = [
    {
        "problem"  : "Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
        "response" : "To determine how much money Janet makes every day at the farmers' market, we need to follow these steps:\n1. **Calculate the total number of eggs laid by the ducks per day.**\n   Janet's ducks lay 16 eggs per day.\n2. **Determine the number of eggs Janet uses each day.**\n   - She eats 3 eggs for breakfast every morning.\n   - She bakes muffins for her friends every day with 4 eggs.\n   So, the total number of eggs used per day is:\n   \\[\n   3 + 4 = 7 \\text{ eggs}\n   \\]\n3. **Calculate the number of eggs Janet sells at the farmers' market each day.**\n   Subtract the number of eggs used from the total number of eggs laid:\n   \\[\n   16 - 7 = 9 \\text{ eggs}\n   \\]\n4. **Determine how much money Janet makes from selling the eggs.**\n   She sells each egg for $2, so the total amount of money she makes is:\n   \\[\n   9 \\times 2 = 18 \\text{ dollars}\n   \\]\nTherefore, the amount of money Janet makes every day at the farmers' market is $\\boxed{18}$."
    },
    {
        "problem"  : "Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
        "response" : "To find out how much Janet makes every day at the farmers' market, we need to follow these steps:\n1. Calculate the total number of eggs laid by the ducks: \n   - Janet's ducks lay 16 eggs per day.\n2. Calculate the total number of eggs consumed by Janet and used for baking:\n   - Janet eats 3 eggs for breakfast.\n   - She bakes 4 eggs for muffins.\n   - Total eggs used: 3 (breakfast) + 4 (baking) = 7 eggs.\n3. Calculate the remaining number of eggs for sale:\n   - Total eggs laid: 16\n   - Eggs used: 7\n   - Remaining eggs: 16 - 7 = 9 eggs\n4. Calculate the total amount of money made at the farmers' market:\n   - Price per egg: $2\n   - Number of eggs sold: 9\n   - Total money made: 9 * $2 = $18\nTherefore, Janet makes $\\boxed{18}$ dollars every day at the farmers' market."
    }
]

# data preprocessing
processed_data = [prepare_input(d["problem"], d["response"], tokenizer=tokenizer, step_token="\n") for d in datas]
input_ids, steps, reward_flags = zip(*processed_data)

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8081/v1"
client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=openai_api_key,
    base_url=openai_api_base,
)
models = client.models.list()
model = models.data[0].id
rewards = client.embeddings.create(
    input=input_ids,
    model=model,
)

step_rewards = derive_step_rewards_vllm(rewards, reward_flags)
print("step_rewards:",step_rewards[0])
print("step_rewards:",step_rewards[1])  
```

# TODO
- Add more results for step-wise RM evaluation methods.
- Adjust the RM architecture to enhance compatibility with vLLM/sglang inference.
- Expand RM use cases by incorporating more types of reasoning tasks.
- Mitigate performance conflicts across different reasoning tasks.

# Contact
If you have any questions, please feel free to reach us at {jujie.he, jiacai.liu}@kunlun-inc.com.

# LICENSE
The community usage of Skywork models require Skywork Community License. The Skywork models support commercial use. If you plan to use the Skywork models or its derivatives for commercial purposes, you must abide by terms and conditions within Skywork Community License.

# DISCLAIMER
We hereby declare that the Skywork models should not be used for any activities that pose a threat to national or societal security or engage in unlawful actions. Additionally, we request users not to deploy the Skywork models for internet services without appropriate security reviews and records. We hope that all users will adhere to this principle to ensure that technological advancements occur in a regulated and lawful environment.

We have done our utmost to ensure the compliance of the data used during the model's training process. However, despite our extensive efforts, due to the complexity of the model and data, there may still be unpredictable risks and issues. Therefore, if any problems arise as a result of using the Skywork open-source model, including but not limited to data security issues, public opinion risks, or any risks and problems arising from the model being misled, abused, disseminated, or improperly utilized, we will not assume any responsibility.

# Citation
If you find our work helpful, please feel free to cite us using the following BibTeX entry:
``` 
@misc{he_2024_16998085,
  author       = {He, Jujie and
                  Wei, Tianwen and
                  Yan, Rui and
                  Liu, Jiacai and
                  Wang, Chaojie and
                  Gan, Yimeng and
                  Tu, Shiwen and
                  Liu, Chris Yuhao and
                  Zeng, Liang and
                  Wang, Xiaokun and
                  Wang, Boyang and
                  Li, Yongcong and
                  Zhang, Fuxiang and
                  Xu, Jiacheng and
                  An, Bo and
                  Liu, Yang and
                  Zhou, Yahui},
  title        = {Skywork-o1 Open Series},
  month        = nov,
  year         = 2024,
  publisher    = {Zenodo},
  version      = {1.0.0},
  doi          = {10.5281/zenodo.16998085},
  url          = {https://doi.org/10.5281/zenodo.16998085},
}
```
