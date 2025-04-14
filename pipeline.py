from re import S
from routellm.controller import Controller
from dataclasses import dataclass

@dataclass
class Config:
    strong_model: str
    weak_model: str 
    router: str
    threshold: float = None  
    api_base: str = 'local'
    only_routing: bool = True

class RoutingQuery(object):    
    def __init__(self, strong_model, weak_model, router, threshold=None):
        self.config = Config(strong_model=strong_model,
                             weak_model=weak_model,
                             router=router,
                             threshold=threshold)
        
        self.controller = Controller(
            strong_model=self.config.strong_model,
            weak_model=self.config.weak_model,
            routers=[self.config.router],
            api_base=self.config.api_base,
            only_routing=self.config.only_routing
        )
        
    
    def route(self, prompt, router=None, threshold=None):
        router = router or self.config.router
        threshold = threshold or self.config.threshold
        
        model = f'router-{router}-{threshold}'
        
        routed_model = self.controller.route(model, prompt) 
        
        return routed_model
    
    
if __name__ == "__main__":
    
    router = RoutingQuery('/home/da02/models/Llama-3.1-8B-Instruct', 
                 '/home/da02/models/Llama-3.2-1B-Instruct',
                 'mf', 
                 0.7)
    
    model_type = router.route(f'What is the capital of France?')
    print(model_type)
    router = RoutingQuery('/home/da02/models/Llama-3.1-8B-Instruct', 
                 '/home/da02/models/Llama-3.2-1B-Instruct',
                 'mf', 
                 0.1)
    print(model_type)
    router = RoutingQuery('/home/da02/models/Llama-3.1-8B-Instruct', 
                 '/home/da02/models/Llama-3.2-1B-Instruct',
                 'sw_ranking', 
                 0.1)
    print(model_type)