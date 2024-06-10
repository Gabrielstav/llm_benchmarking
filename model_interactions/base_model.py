from abc import ABC, abstractmethod
from typing import Any, Dict
from model_benchmarking.containers.data_container import ModelResponse, ApiResponse

class ModelInteraction(ABC):
    """
    Abstract base class for model interactions.

    Methods:
        prepare_query(input_data): Prepares the data for sending to the model.
        send_query(prepared_data): Sends the prepared data to the model.
        process_response(response) -> ApiResponse: Processes the response from the model.
        instantiate_dataclasses(parsed_data): Instantiates dataclasses with the parsed data response from the model.
    """

    @abstractmethod
    def prepare_query(self, input_data: Dict) -> Dict:
        """
        Prepare the data for sending to the model.

        Args:
            input_data (Any): The input data to be prepared for the model.
        Returns:
            Dict: The prepared data to be sent to the model.
        """
        pass

    @abstractmethod
    async def send_query(self, prepared_data: Dict) -> Dict:
        """
        Send the prepared data to the model.

        Args:
            prepared_data (Dict): The prepared data to be sent to the model.
        Returns:
            Dict: The response from the model.
        """
        pass

    @abstractmethod
    def process_response(self, response: Any) -> ApiResponse:
        """
        Process the response from the model.

        Args:
            response (Dict): The response from the model.
        Returns:
            ApiResponse: The processed response.
        """
        pass

    @abstractmethod
    def instantiate_dataclasses(self, parsed_data: ApiResponse) -> ModelResponse:
        """
        Instantiate DomainAnalysis, (optionally ResponseMetadata) and ModelResponse dataclasses with parsed data response from the model.

        Args:
            parsed_data (ApiResponse): The parsed data response from the model.
        Returns:
            Any: The instantiated dataclasses.
        """
        pass