from contextvars import ContextVar
import aiohttp
from typing import Dict, Any
from model_interactions.base_model import ModelInteraction
from containers.data_container import ApiResponse, DomainAnalysis, ModelResponse

# use context var for domain name for smaller models to instantiate dataclasses,
# since these models do not return the domain name in their response 
domain_name_context = ContextVar("domain_name")

class FineTunedInteraction(ModelInteraction):

    def __init__(self, config):
        self.config = config 

    def prepare_query(self, input_data: Dict) -> Dict:
        domain_name_context.set(input_data)
        return {
            "inputs": input_data, 
            "options": {
                "wait_for_model": True, 
                "use_cache": True
            }
        }
    
    async def send_query(self, prepared_data: Dict) -> Dict: 
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-type": "application/json"
            "wait_for_model=True, use_cache=True"
        }

        async with aiohttp.ClientSession() as session: 
            async with session.post(self.config.end_point, json=prepared_data, headers=headers) as response: 
                if response.status != 200: 
                    response_text = await response.text()
                    raise Exception(f"API Error: {response.status} - {response_text}")
                
                response_data = await response.json()

                return response_data  
            
    def process_response(self, response: Any) -> ApiResponse:
        try:
            # first dictionary is always the highest-confidence category 
            highest_confidence_response = response[0][0]  

            label = highest_confidence_response["label"].lower()
            label = label.replace("malware", "malicious") 
            score = highest_confidence_response["score"]

            domain_name = domain_name_context.get()
            domain_name = domain_name.strip(" \n[]\t\r")  

            processed_data = {
                "domain_name": domain_name,
                "label": label,
                "score": score
            }

            return ApiResponse(data=processed_data, metadata=None)

        except (IndexError, KeyError, TypeError) as e:
            raise ValueError(f"Failed to process model response for model {self.config.model_name}: {e}")
        
    def instantiate_dataclasses(self, parsed_data: ApiResponse) -> ModelResponse | Any:
        try:
            # instantiate DomainAnalysis class 
            analysis_results = [DomainAnalysis(
                domain_name=parsed_data.data["domain_name"],
                category=parsed_data.data["label"], 
                confidence_score=parsed_data.data["score"])]
            
            # instantiate ModelResponse class
            return ModelResponse(
                model_name=self.config.model_name,
                analysis_results=analysis_results,
                metadata=None 
            )

        except Exception as e:
            print("Error in dataclass instantiation:", e)
            return None