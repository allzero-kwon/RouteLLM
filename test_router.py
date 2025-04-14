import os
import json
from routellm.controller import Controller
from routellm.evals.benchmarks import GSM8K
import pandas as pd 
from collections import defaultdict

output_dir='./results'
strong_model = "/home/da02/models/Llama-3.1-8B-Instruct"
weak_model = "/home/da02/models/Llama-3.2-1B-Instruct"

print(f'create controller')

# router = 'causal_llm'
# router_idx=3


 
router = 'bert'
router_idx=2 # set mf=0, sw_ranking=1, bert=2, causal_llm=3

thresholds = {
    # mf, sw_ranking, bert, causal llm 
    '0.01' : [0.38161, 0.22274, 0.81778, 0.53434],
    '0.03' : [0.32544, 0.22154, 0.74842, 0.41692],
    '0.05' : [0.29185, 0.22092, 0.68507, 0.36443],
    '0.1' : [0.24034, 0.21995, 0.59491, 0.29102],
    '0.3' : [0.15609, 0.21796, 0.46514, 0.15774],
    '0.5' : [0.11593, 0.21647, 0.4066, 0.0962]
    
}

controller = Controller(
  routers=[router],
  strong_model=strong_model,
  weak_model=weak_model,
  api_base='local',
  only_routing=True
)

def load_data(task):
    return pd.read_json(f"./cot_evals/{task}.jsonl", lines=True)
tasks = ['gsm8k_cot_llama_3.1_instruct','asdiv_cot_qwq','math_cot_llama_3.1_instruct' ] #
from tqdm import tqdm

final = {}
for i, threshold in thresholds.items():
    final[i] = {}
    for task in tasks:
        all_data = load_data(task)
        datasets = []
        count = defaultdict(int)
        print(f'output : {router}_{task}_{i}_test.jsonl')
        f = open(f'{router}_{task}_{i}_test.jsonl', 'w')
        for prompt in tqdm(all_data['prompt']):
            routed_model1 = controller.chat.completions.create(
                model=f"router-{router}-{threshold[router_idx]}", # bert=2
                messages=[
                    {"role": "user", "content": prompt}
                ],
                only_routing=True
            )
             
            count[router] += 1 if routed_model1 == strong_model else 0
            
            if (count[router]) % 10 == 0 :
                print(f'{router} ', routed_model1, '(',  count[router], ')')
                
            datasets.append({"prompt": prompt, f"{router}_model" : routed_model1})
            f.write(json.dumps({"prompt": prompt, f"{router}_model" : routed_model1}) + '\n')
        
        count[f'{router}_win_ratio'] = count[router] / len(datasets)
        final[i][task] = count
        
        f.close()
        print(f'count : {count}')
        print(f'datasets : {len(datasets)}')

with open('final_results.json', 'w') as f:
    json.dump(final, f, indent=4)
    print(json.dumps(final, indent=4))