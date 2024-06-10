import json
import aiohttp
import asyncio
from typing import Dict, Any
from model_interactions.base_model import ModelInteraction
from containers.data_container import ApiResponse, DomainAnalysis, ModelResponse, ResponseMetadata

# TODO API error 429 for Anthropic - rate limit exceeded. Need to upgrade subscription for actual processing. 
#      can automate this with exponential backoff and indexing into output file, to send new payloads corresponding to correct index in output data
#      also should set option to delay API requests, evenly spacing them out with X delay. 

class AnthropicInteraction(ModelInteraction):

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
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.config.api_key,
            "anthropic-version": "2023-06-01",
        }

        retry_timings = [2, 4, 12, 16, 20, 30]

        # need to manually serialize the data for Anthropic requests to work 
        json_data = json.dumps(prepared_data) 

        async with aiohttp.ClientSession() as session:
            for retry in retry_timings:
                try:
                    async with session.post(self.config.end_point, data=json_data, headers=headers) as response:
                        if response.status == 429:
                            retry_after = response.headers.get("Retry-After", retry)
                            print(f"API error 429, rate limit exceeded. Retrying after {retry_after} seconds.")
                            await asyncio.sleep(int(retry_after))
                            continue  # retry logic

                        if response.status != 200:
                            response_text = await response.text()
                            print(f"API error {response.status}: {response_text}")
                            raise Exception(f"API Error: {response.status} - {response_text}")

                        response_data = await response.json()
                        if "error" in response_data:
                            raise Exception(f"API Error: {response_data['error']['message']}")
                        return response_data

                except Exception as e:
                    print(f"Exception during request: {e}")
                    if retry == retry_timings[-1]:
                        raise  # Raise the exception if it's the last retry

        raise Exception("API request failed after multiple retries")
    
    def process_response(self, response: Any) -> ApiResponse:
        try:
            # extract text from content list 
            content = next(item for item in response["content"] if item["type"] == "text")["text"]
            json_start_idx = content.find("{")
            json_end_idx = content.rfind("}") + 1
            if json_start_idx == -1 or json_end_idx == -1:
                raise ValueError("JSON object not found in response content.")

            cleaned_content = content[json_start_idx:json_end_idx].strip()
            parsed_data = json.loads(cleaned_content)

            # get metadata
            usage = response["usage"]
            metadata = {
                "prompt_tokens": usage["input_tokens"],
                "completion_tokens": usage["output_tokens"],
                "total_tokens": usage["input_tokens"] + usage["output_tokens"]
            }

            return ApiResponse(data=parsed_data, metadata=metadata)

        except (KeyError, ValueError, json.JSONDecodeError) as e:
            raise ValueError(f"Failed to parse model {self.config.model_name} response: {e}")
        
    def instantiate_dataclasses(self, parsed_data: ApiResponse) -> ModelResponse | None:
        try:

            analysis_results = []

            # batch instantiation of DomainAnalysis class 
            if not self.config.single_mode:
                for result in parsed_data.data["analysis_results"]:
                    analysis = DomainAnalysis(
                        domain_name=result["domain_name"],
                        category=result["category"],
                        confidence_score=int(result["confidence_score"]),
                        reasons=result["reasons"]
                    )
                    analysis_results.append(analysis)
            
            # single mode domain analysis instantiation (not wrapped in results)
            else:   
                analysis_results = [DomainAnalysis(
                    domain_name=parsed_data.data["domain_name"],
                    category=parsed_data.data["category"],
                    confidence_score=parsed_data.data["confidence_score"],
                    reasons=parsed_data.data["reasons"])]    
    
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
            return None