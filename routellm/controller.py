from collections import defaultdict
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import requests
import pandas as pd
from litellm import acompletion, completion
from tqdm import tqdm
import torch

from routellm.routers.routers import ROUTER_CLS

# Default config for routers augmented using golden label data from GPT-4.
# This is exactly the same as config.example.yaml.
GPT_4_AUGMENTED_CONFIG = {
    "sw_ranking": {
        "arena_battle_datasets": [
            "lmsys/lmsys-arena-human-preference-55k",
            "routellm/gpt4_judge_battles",
        ],
        "arena_embedding_datasets": [
            "routellm/arena_battles_embeddings",
            "routellm/gpt4_judge_battles_embeddings",
        ],
    },
    "causal_llm": {"checkpoint_path": "routellm/causal_llm_gpt4_augmented"},
    "bert": {"checkpoint_path": "routellm/bert_gpt4_augmented"},
    "mf": {"checkpoint_path": "routellm/mf_gpt4_augmented"},
}


    
def chat(req, tokenizer, model):
    history = []
    for msg in req['messages']:
        history.append((msg['role'], msg['content']))
    
    user_input = [msg['content'] for msg in req['messages'] if msg['role'] == "user"][-1]
    messages = req['messages']
    
    def build_prompt(messages):
        prompt = ""
        for m in messages:
            if m['role'] == "user":
                prompt += f"<|user|>\n{m['content']}\n"
            elif m['role'] == "assistant":
                prompt += f"<|assistant|>\n{m['content']}\n"
        prompt += "<|assistant|>\n"
        return prompt

    input_text = build_prompt(messages)

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=2048,
        temperature=0.7,
        top_p=0.95,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    
    return {
        "id": "chatcmpl-fakeid",
        "object": "chat.completion",
        "model": req['model'],
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": response
            },
            "finish_reason": "stop"
        }],
    }




class RoutingError(Exception):
    pass


@dataclass
class ModelPair:
    strong: str
    weak: str


class Controller:
    def __init__(
        self,
        routers: list[str],
        strong_model: str,
        weak_model: str,
        config: Optional[dict[str, dict[str, Any]]] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        progress_bar: bool = False,
    ):
        self.model_pair = ModelPair(strong=strong_model, weak=weak_model)
        self.routers = {}
        self.api_base = api_base
        self.api_key = api_key
        self.model_counts = defaultdict(lambda: defaultdict(int))
        self.progress_bar = progress_bar
        
        if api_base == "local":
            print(f'Load weak : {weak_model}')
            self._weak_tokenizer = AutoTokenizer.from_pretrained(weak_model, trust_remote_code=True)
            self._weak_model = AutoModelForCausalLM.from_pretrained(
                weak_model,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            self._weak_model.eval()
            print(f'Load strong : {strong_model}')
                        
            self._strong_tokenizer = AutoTokenizer.from_pretrained(strong_model, trust_remote_code=True)
            self._strong_model = AutoModelForCausalLM.from_pretrained(
                strong_model,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            self._strong_model.eval()
            
            

        if config is None:
            config = GPT_4_AUGMENTED_CONFIG

        router_pbar = None
        if progress_bar:
            router_pbar = tqdm(routers)
            tqdm.pandas()

        for router in routers:
            if router_pbar is not None:
                router_pbar.set_description(f"Loading {router}")
            self.routers[router] = ROUTER_CLS[router](**config.get(router, {}))

        # Some Python magic to match the OpenAI Python SDK
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(
                create=self.completion, acreate=self.acompletion
            )
        )

    def _validate_router_threshold(
        self, router: Optional[str], threshold: Optional[float]
    ):
        if router is None or threshold is None:
            raise RoutingError("Router or threshold unspecified.")
        if router not in self.routers:
            raise RoutingError(
                f"Invalid router {router}. Available routers are {list(self.routers.keys())}."
            )
        if not 0 <= threshold <= 1:
            raise RoutingError(
                f"Invalid threshold {threshold}. Threshold must be a float between 0.0 and 1.0."
            )

    def _parse_model_name(self, model: str):
        _, router, threshold = model.split("-", 2)
        try:
            threshold = float(threshold)
        except ValueError as e:
            raise RoutingError(f"Threshold {threshold} must be a float.") from e
        if not model.startswith("router"):
            raise RoutingError(
                f"Invalid model {model}. Model name must be of the format 'router-[router name]-[threshold]."
            )
        return router, threshold

    def _get_routed_model_for_completion(
        self, messages: list, router: str, threshold: float
    ):
        # Look at the last turn for routing.
        # Our current routers were only trained on first turn data, so more research is required here.
        prompt = messages[-1]["content"]
        routed_model = self.routers[router].route(prompt, threshold, self.model_pair)
        self.model_counts[router][routed_model] += 1

        return routed_model

    # Mainly used for evaluations
    def batch_calculate_win_rate(
        self,
        prompts: pd.Series,
        router: str,
    ):
        self._validate_router_threshold(router, 0)
        router_instance = self.routers[router]
        if router_instance.NO_PARALLEL and self.progress_bar:
            return prompts.progress_apply(router_instance.calculate_strong_win_rate)
        elif router_instance.NO_PARALLEL:
            return prompts.apply(router_instance.calculate_strong_win_rate)
        else:
            return prompts.parallel_apply(router_instance.calculate_strong_win_rate)

    def route(self, prompt: str, router: str, threshold: float):
        self._validate_router_threshold(router, threshold)

        return self.routers[router].route(prompt, threshold, self.model_pair)

    # Matches OpenAI's Chat Completions interface, but also supports optional router and threshold args
    # If model name is present, attempt to parse router and threshold using it, otherwise, use the router and threshold args
    def completion(
        self,
        *,
        router: Optional[str] = None,
        threshold: Optional[float] = None,
        **kwargs,
    ):
        if "model" in kwargs:
            router, threshold = self._parse_model_name(kwargs["model"])

        self._validate_router_threshold(router, threshold)
        kwargs["model"] = self._get_routed_model_for_completion(
            kwargs["messages"], router, threshold
        )
        
        if kwargs['only_routing'] == True :
            return kwargs["model"]
                
        def local_completion(model, messages):
            headers = {
                "Authorization": "Bearer local",
                "Content-Type": "application/json"
            }
            url = f"{model}/chat/completions"
            payload = {
                "model": "router-mf-0.11593",
                "messages": [
                    {"role": "user", "content": messages}
                ]
            }
            response = requests.post(url, headers=headers, json=payload)
            response_data = response.json()
            
            return response_data
        if 'http://' in kwargs["model"]:
            # Handle the case where the model name is a URL
            model = kwargs["model"]
            messages = kwargs["messages"][0]['content']
            response_data = local_completion(model, messages)
            return response_data
        elif self.api_base == "local":
            if 'QwQ' in kwargs["model"] or 'Llama-3.1-8B-Instruct' in kwargs["model"]:
                return chat(kwargs, self._strong_tokenizer, self._strong_model)
            else : 
                return chat(kwargs, self._weak_tokenizer, self._weak_model)
        else : 
            return completion(api_base=self.api_base, api_key=self.api_key, **kwargs)

    # Matches OpenAI's Async Chat Completions interface, but also supports optional router and threshold args
    async def acompletion(
        self,
        *,
        router: Optional[str] = None,
        threshold: Optional[float] = None,
        **kwargs,
    ):
        if "model" in kwargs:
            router, threshold = self._parse_model_name(kwargs["model"])

        self._validate_router_threshold(router, threshold)
        kwargs["model"] = self._get_routed_model_for_completion(
            kwargs["messages"], router, threshold
        )
        return await acompletion(api_base=self.api_base, api_key=self.api_key, **kwargs)
