import json
import aiohttp
from typing import Dict, Any
from model_benchmarking.model_interactions.base_model import ModelInteraction
from model_benchmarking.containers.data_container import ApiResponse, DomainAnalysis, ModelResponse


class LlamaInteraction(ModelInteraction):    
    def __init__(self, config):
        self.config = config 

    def prepare_query(self, input_data: Dict) -> Dict: 
        return { 
            "inputs": input_data,
            "parameters": {  
                "max_new_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "stream": False,
                "return_full_text": False
            },
            "options": {
                "use_cache": False,
            }
        }

    async def send_query(self, prepared_data: Dict) -> Dict:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json", 
            "Authorization": f"Bearer {self.config.api_key}",
            "Response-Format": "json_object"
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(self.config.end_point, headers=headers, json=prepared_data) as response:
                response_text = await response.text()
                
                if response.status != 200:
                    print(f"HTTP Error: {response.status}")
                    return {"error": "Failed to process the request", "details": response_text}

                try:
                    response_data = json.loads(response_text)
                except json.JSONDecodeError:
                    print("Failed to decode JSON from response")
                    return {"error": "Failed to decode JSON", "details": response_text}
        
                return response_data
            
    def process_response(self, response: Any) -> ApiResponse:
        print(f"Response data for Llama: {response}")

        if not response or 'error' in response:
            print("Empty or error response received.")
            raise ValueError("Received an empty or error response from the API.")
        
        # TODO 
        # Use "}" as stop token, but this needs to be the last bracket in json string 
        # alternatively I can use markdown backticks but unsure how consistent that formatting is 
        # as a response from llama. 

        # Check for empty generated_text
        content = response[0].get("generated_text", "")
        if not content:
            print("Received empty generated_text in the response.")
            raise ValueError("Generated text is empty in the response.")

        print(f"Initial Content: {content}")

        # Find start/end of JSON content
        json_start_idx = content.find("{")
        if json_start_idx == -1:
            raise ValueError("JSON start character '{' not found in response content.")
        json_end_idx = content.rfind("}") + 1

        # Extract JSON content
        try:
            json_str = content[json_start_idx:json_end_idx]
            json_data = json.loads(json_str)
            print("Extracted JSON Data:", json_data)
        except json.JSONDecodeError as e:
            print("Error decoding JSON:", e)
            raise ValueError("Failed to decode JSON from the content.")

        # Batch mode has analysis_results key
        if not self.config.single_mode:
            analysis_results = json_data.get("analysis_results")
            if analysis_results is None:
                raise ValueError("Analysis results not found in JSON data.")
            return ApiResponse(data=analysis_results, metadata=None)
        else:
            # Single mode does not, return data as is
            return ApiResponse(data=json_data, metadata=None)

    def instantiate_dataclasses(self, parsed_data: ApiResponse) -> ModelInteraction | Any:
        try:
            analysis_results = []

            # batch instantiation of DomainAnalysis class 
            # looping over each result
            if not self.config.single_mode:
                for result in parsed_data.data:
                    print(f"Items in parsed_data: {result}")
                    analysis = DomainAnalysis(
                        domain_name=result["domain_name"],
                        category=result["category"],
                        confidence_score=int(result["confidence_score"]),
                        reasons=result["reasons"]
                    )
                    analysis_results.append(analysis)
                    
            else:
            # single mode instantiation of DomainAnalysis class 
                analysis_results = [DomainAnalysis(
                    domain_name=parsed_data.data["domain_name"],
                    category=parsed_data.data["category"], 
                    confidence_score=parsed_data.data["confidence_score"],
                    reasons=parsed_data.data["reasons"])]
            
            # instantiate ModelResponse class
            return ModelResponse(
                model_name=self.config.model_name,
                analysis_results=analysis_results,
                metadata=None 
            )

        except Exception as e:
            print("Error in dataclass instantiation:", e)
            return None