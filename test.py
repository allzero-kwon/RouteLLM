import os
import json
from routellm.controller import Controller
from routellm.evals.benchmarks import GSM8K
import pandas as pd 
from collections import defaultdict

controller = Controller(
  routers=["mf"],
  strong_model="/home/da02/models/Llama-3.1-8B-Instruct",
  weak_model="/home/da02/models/Llama-3.2-1B-Instruct",
  api_base='local'
)
 
all_data = pd.read_csv('./routellm/evals/gsm8k/gsm8k_responses.csv')
contaminated_prompts = pd.read_json('./routellm/evals/gsm8k/contaminated_prompts.jsonl', lines=True)['eval_prompt'].tolist()
all_data=all_data[~all_data['prompt'].isin(contaminated_prompts)]

datasets = []
count = defaultdict(int)

for prompt in all_data['prompt']:
        
    routed_model = controller.chat.completions.create(
        model="router-mf-0.31",
        messages=[
            {"role": "user", "content": prompt}
        ],
        only_routing=True
    )
    
    print(routed_model)
    count[routed_model] += 1
    datasets.append({"prompt": prompt, "model" : routed_model})
    
print(f'datasets : {len(datasets)}')
