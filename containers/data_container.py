from typing import Optional, List
from pathlib import Path 
from pydantic import BaseModel

# TODO:
# Add validation for non-empty metadata, confidence score, model name = config model name etc 

# return type from process_response methods  
class ApiResponse:
    """
    Class to hold the API response data.

    Attributes:
        data (Any): The main data from the API response.
        metadata (Optional[Dict]): Additional metadata from the API response.
    """
    def __init__(self, data, metadata=None):
        self.data = data
        self.metadata = metadata if metadata else {}

class DomainAnalysis(BaseModel):
    """
    Pydantic model to represent the analysis of a domain.

    Attributes:
        domain_name (str): The name of the domain.
        category (str): The category of the domain.
        confidence_score (float): The confidence score of the analysis.
        reasons (Optional[List[str]]): The reasons for the analysis classification.
    """

    domain_name: str
    category: str
    confidence_score: float
    reasons: Optional[List[str]] = None 

class ResponseMetadata(BaseModel):
    """
    Pydantic model to represent metadata from the API response.

    Attributes:
        prompt_tokens (int): The number of tokens in the prompt.
        completion_tokens (int): The number of tokens in the completion.
        total_tokens (int): The total number of tokens used.
        time_taken (Optional[int]): The time taken for the response.
        prompt_used (Optional[str]): The prompt used in the request.
        finish_reason (Optional[str]): The reason for finishing the request.
        logprobs (Optional[str]): Log probabilities for the response.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    time_taken: Optional[int] = None 
    prompt_used: Optional[str] = None  
    finish_reason: Optional[str] = None  
    logprobs: Optional[str] = None 
    

class ModelResponse(BaseModel):
    """
    Pydantic model to represent the complete response from the model.

    Attributes:
        model_name (str): The name of the model.
        analysis_results (List[DomainAnalysis]): The list of domain analysis results.
        metadata (Optional[ResponseMetadata]): Additional metadata from the response.
    """
    
    model_name: str
    analysis_results: List[DomainAnalysis]
    metadata: Optional[ResponseMetadata] = None
    output_data_path: Optional[Path] = None 

