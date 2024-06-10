import json
import aiohttp
from typing import Dict, Any
from model_interactions.base_model import ModelInteraction
from containers.data_container import ApiResponse, DomainAnalysis, ModelResponse, ResponseMetadata

class DeepSeekInteraction(ModelInteraction):
    def __init__(self, config):
        self.config = config

    def prepare_query(self, input_data: Dict) -> Dict:

        return {
            "model": self.config.model_name, 
            "messages": [
                {"role": "user", "content": input_data}
            ],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "stream": False
        }
    
    async def send_query(self, prepared_data: Dict) -> Dict:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.config.api_key}"
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(self.config.end_point, headers=headers, json=prepared_data) as response:
                response_text = await response.text()  
                print(f"Response: {response_text}")
                if response.status != 200:
                    raise Exception(f"API Error: {response.status}")
                response_data = json.loads(response_text)
                if "error" in response_data:
                    error_msg = response_data["error"]["message"]
                    raise Exception(f"API Error: {error_msg}")
                return response_data
            
    def process_response(self, response: Any) -> ApiResponse:
        max_retries = 3 
        try:
            if self.config.single_mode:
                results = [response["choices"][0]]
            else:
                results = response["choices"]

            parsed_responses = []
            for result in results:
                retries = 0
                while retries < max_retries:
                    try:
                        content = result["message"]["content"]

                        # find start/end of json 
                        json_start_idx = content.find("{")
                        json_end_idx = content.rfind("}") + 1  
                        if json_start_idx == -1 or json_end_idx == -1:
                            raise ValueError("JSON start or end character not found in response content.")
                        cleaned_content = content[json_start_idx:json_end_idx]
                        json_data = json.loads(cleaned_content)
                        parsed_responses.append(json_data)
                        break  # exit retry loop on success

                    except json.decoder.JSONDecodeError as e:
                        print(f"Retry {retries + 1}/{max_retries} failed with error: {str(e)}")
                        retries += 1
                        if retries >= max_retries:
                            raise  # reraise the exception after last retry

            # extract metadata
            usage = response.get("usage", {})
            metadata = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0)
            }

            return ApiResponse(data=parsed_responses, metadata=metadata)
        
        except Exception as e:
            print(f"Error processing response: {e}")
            raise

        
    def instantiate_dataclasses(self, parsed_data: ApiResponse) -> ModelResponse | Any:
        try:
            # batch processing to instantiate DomainAnalysis class 
            analysis_results = []
            if not self.config.single_mode:
                for item in parsed_data.data:
                    for result in item["analysis_results"]:
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