import json
import aiohttp
from typing import Dict, Any
from model_benchmarking.model_interactions.base_model import ModelInteraction
from model_benchmarking.containers.data_container import ApiResponse, DomainAnalysis, ModelResponse, ResponseMetadata

# TODO: 
# Fix extra data and missing data in gpt4-preview-1164 
# Need to initialize with constructed prompt 
# and extract the instantiation of the dataclasses 
# so the logic is centralize and I don't need to manually change the method across subclasses 
# if changing specification. 
# I need the ModelResponse to contain prompt used, time taken, etc 

class OpenAIInteraction(ModelInteraction):
    def __init__(self, config):
        self.config = config  

    def prepare_query(self, input_data: Dict) -> Dict:
        return {
            "model": self.config.model_name,  
            "messages": [{"role": "user", "content": input_data}],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature
        }
    
    async def send_query(self, prepared_data: Dict) -> Dict:
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
                "OpenAI-Organization": self.config.organization_id
            }

            async with session.post(self.config.end_point, json=prepared_data, headers=headers) as response:
                if response.status != 200:
                    raise Exception(f"API Error: {response.status}")
                response_data = await response.json()
                if "error" in response_data:
                    error_msg = response_data["error"]["message"]
                    raise Exception(f"API Error: {error_msg}")
                return response_data
            
    def process_response(self, response: Any) -> ApiResponse:
        try:
            if self.config.single_mode:
                # single mode contains one analysis (this gets the first even if multiple are sent)
                results = [response["choices"][0]]
            else:
                # get all in batch mode 
                results = response["choices"]

            parsed_responses = []
            for result in results:
                # extract the content which contains the JSON string
                content = result["message"]["content"]

                # find JSON start and strip chars 
                json_start_idx = content.find("{")
                if (json_start_idx == -1):
                    raise ValueError("JSON start character '{' not found in response content.")
                
                cleaned_content = content[json_start_idx:]
                
                # remove all backticks
                cleaned_content = cleaned_content.replace("`", "").strip()

                # find the end of JSON if extra data is an issue
                json_end_idx = cleaned_content.rfind("}")
                if json_end_idx != -1:
                    cleaned_content = cleaned_content[:json_end_idx + 1]

                print(f"Cleaned content: {cleaned_content}")

                json_data = json.loads(cleaned_content)
                parsed_responses.append(json_data)

            # extract metadata
            usage = response.get("usage", {})
            metadata = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0)
            }

            return ApiResponse(data=parsed_responses, metadata=metadata)

        except (KeyError, IndexError, json.JSONDecodeError) as e:
            raise ValueError(f"Failed to parse model {self.config.model_name} response: {e}")

        
    def instantiate_dataclasses(self, parsed_data: ApiResponse) -> ModelInteraction | Any: 
        try:
            # batch processing to instantiate DomainAnalysis class 
            analysis_results = []
            if not self.config.single_mode:
                for item in parsed_data.data:
                    for result in item['analysis_results']:
                        analysis = DomainAnalysis(
                            domain_name=result["domain_name"],
                            category=result["category"],
                            confidence_score=int(result["confidence_score"]),
                            reasons=result["reasons"]
                        )
                        analysis_results.append(analysis)
            else:
                # single mode processing to instantiate DomainAnalysis class 
                analysis_results = [
                    DomainAnalysis(
                    domain_name=item["domain_name"],
                    category=item["category"],
                    confidence_score=int(item["confidence_score"]),
                    reasons=item["reasons"]
                )
                for item in parsed_data.data  
            ]


            # instantiate ResponseMetadata class
            response_metadata = ResponseMetadata(
                prompt_tokens=parsed_data.metadata["prompt_tokens"],
                completion_tokens=parsed_data.metadata["completion_tokens"],
                total_tokens=parsed_data.metadata["total_tokens"]
            )

            # instantiate ModelResponse class
            return ModelResponse(
                model_name=self.config.model_name,
                analysis_results=analysis_results,
                metadata=response_metadata
            )

        except Exception as e:
            print("Error in dataclass instantiation:", e)