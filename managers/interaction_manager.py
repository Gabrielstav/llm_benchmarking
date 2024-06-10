import pandas as pd
import asyncio
import json
from typing import List
from configs.config_loader import ModelConfig, get_validated_data
from data_processing.save_response import SaveResponse
from data_processing.construct_prompt import ConstructPrompt
from model_interactions.llama import LlamaInteraction
from model_interactions.open_ai import OpenAIInteraction
from model_interactions.anthropic import AnthropicInteraction
from model_interactions.deepseek import DeepSeekInteraction
from model_interactions.finetuned_bert import FineTunedInteraction

    
class ModelInteractionManager:
    """
    Class to manage interactions with multiple models, process the responses, and save the results.

    Attributes:
        configurations (List[ModelConfig]): A list of configurations for the models.
        interactions (List[ModelInteraction]): A list of model interaction instances.
        response_savers (List[SaveResponse]): A list of response saver instances.
    """

    def __init__(self, configurations: List[ModelConfig], buffer_size: int):
        self.configurations = configurations
        self.interactions = [self._create_model_interaction(config) for config in configurations]
        self.response_savers = [SaveResponse(config.model_name, buffer_size=buffer_size, config=config) for config in configurations]

    async def run_interactions(self):
        """
        Runs the interactions for all configured models concurrently.
        """
        tasks = [
            self._run_single_interaction(config, interaction, response_saver)
            for config, interaction, response_saver in zip(self.configurations, self.interactions, self.response_savers)
        ]
        await asyncio.gather(*tasks)

    async def _run_single_interaction(self, config, interaction, response_saver):
        """
        Runs the interaction for a single model configuration.

        Args:
            config (ModelConfig): The configuration for the model.
            interaction (ModelInteraction): The model interaction instance.
            response_saver (SaveResponse): The response saver instance.
        """
        validated_data = pd.read_csv(get_validated_data())
        await self._process_batches(config, interaction, response_saver, validated_data)
        response_saver._flush_buffer()

    async def _process_batches(self, config, interaction, response_saver, validated_data):
        """
        Processes the data in batches for a single model configuration.

        Args:
            config (ModelConfig): The configuration for the model.
            interaction (ModelInteraction): The model interaction instance.
            response_saver (SaveResponse): The response saver instance.
            validated_data (pd.DataFrame): The validated data to process.
        """
        total_batches = self._calculate_total_batches(config)
        tasks = [
            self._process_single_batch(batch_num, config, validated_data, interaction, response_saver)
            for batch_num in range(total_batches)
        ]
        await asyncio.gather(*tasks)

    async def _process_single_batch(self, batch_num, config, validated_data, interaction, response_saver):
        """
        Processes a single batch of data for a model configuration.

        Args:
            batch_num (int): The batch number.
            config (ModelConfig): The configuration for the model.
            validated_data (pd.DataFrame): The validated data to process.
            interaction (ModelInteraction): The model interaction instance.
            response_saver (SaveResponse): The response saver instance.
        """
        start_index, end_index = self._get_batch_indices(batch_num, config, len(validated_data))
        if start_index < len(validated_data):
            batch_data = validated_data.iloc[start_index:end_index]
            await self._process_and_save_batch(batch_data, config, interaction, response_saver)

    async def _process_and_save_batch(self, batch_data, config, interaction, response_saver):
        """
        Processes and saves a single batch of data.

        Args:
            batch_data (pd.DataFrame): The batch data to process.
            config (ModelConfig): The configuration for the model.
            interaction (ModelInteraction): The model interaction instance.
            response_saver (SaveResponse): The response saver instance.
        """
        prompt = self._construct_prompt(batch_data, config)
        prepared_data = interaction.prepare_query(prompt)
        response = await self._send_and_process_response(prepared_data, interaction)
        response_saver.add_data(response)

    def _construct_prompt(self, batch_data, config):
        """
        Constructs the prompt for the model based on the batch data and configuration.

        Args:
            batch_data (pd.DataFrame): The batch data.
            config (ModelConfig): The configuration for the model.

        Returns:
            str: The constructed prompt.
        """
        prompt_constructor = ConstructPrompt(domains_data=batch_data, config=config)
        return prompt_constructor.construct_prompt()

    async def _send_and_process_response(self, prepared_data, interaction):
        """
        Sends the prepared data to the model and processes the response.

        Args:
            prepared_data (dict): The prepared data to send to the model.
            interaction (ModelInteraction): The model interaction instance.

        Returns:
            ModelResponse: The processed response from the model.
        """
        response = await self._send_with_retries(prepared_data, interaction)
        return interaction.instantiate_dataclasses(response)

    async def _send_with_retries(self, prepared_data, interaction):
        """
        Sends the prepared data to the model with retries on failure.

        Args:
            prepared_data (dict): The prepared data to send to the model.
            interaction (ModelInteraction): The model interaction instance.
        Returns:
            ApiResponse: The processed response from the model.
        Raises:
            Exception: If the request fails after the maximum number of retries.
        """
        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                response = await interaction.send_query(prepared_data)
                if "error" in response:
                    print(f"Error in response: {response}")
                return interaction.process_response(response)
            except (json.decoder.JSONDecodeError, KeyError) as e:
                await self._handle_retryable_error(attempt, max_retries, retry_delay, e)
            except Exception as e:
                await self._handle_non_retryable_error(attempt, max_retries, retry_delay, e)

    async def _handle_retryable_error(self, attempt, max_retries, retry_delay, error):
        """
        Handles retryable errors during sending with retries.

        Args:
            attempt (int): The current attempt number.
            max_retries (int): The maximum number of retries.
            retry_delay (int): The delay between retries.
            error (Exception): The error that occurred.
        Raises:
            Exception: If the maximum number of retries is reached.
        """
        print(f"Other error on attempt {attempt+1}: {error}")
        if attempt < max_retries - 1:
            await asyncio.sleep(retry_delay)
        else:
            raise Exception(f"Unexpected error during interaction: {error}")


    async def _handle_non_retryable_error(self, attempt, max_retries, retry_delay, error):
        """
        Handles non-retryable errors during sending with retries.

        Args:
            attempt (int): The current attempt number.
            max_retries (int): The maximum number of retries.
            retry_delay (int): The delay between retries.
            error (Exception): The error that occurred.

        Raises:
            Exception: If the maximum number of retries is reached.
        """
        print(f"Other error on attempt {attempt+1}: {error}")
        if attempt < max_retries - 1:
            await asyncio.sleep(retry_delay)
        else:
            raise Exception(f"Unexpected error during interaction: {error}")

    def _calculate_total_batches(self, config):
        """
        Calculates the total number of batches to process based on the configuration.

        Args:
            config (ModelConfig): The configuration for the model.
        Returns:
            int: The total number of batches.
        """
        return (config.total_domains + config.batch_size - 1) // config.batch_size

    def _get_batch_indices(self, batch_num, config, data_length):
        """
        Gets the start and end indices for a batch based on the batch number and configuration.

        Args:
            batch_num (int): The batch number.
            config (ModelConfig): The configuration for the model.
            data_length (int): The length of the data.
        Returns:
            tuple: The start and end indices for the batch.
        """
        start_index = config.start_index + batch_num * config.batch_size
        end_index = min(start_index + config.batch_size, config.start_index + config.total_domains, data_length)
        return start_index, end_index

    def _create_model_interaction(self, model_config):
        """
        Creates the appropriate model interaction instance based on the model configuration.

        Args:
            model_config (ModelConfig): The configuration for the model.
        Returns:
            ModelInteraction: The appropriate model interaction instance.
        Raises:
            ValueError: If the model name is unsupported.
        """
        print(f"Creating interaction for model: {model_config.model_name}")
        model_key = model_config.model_name.lower()
        
        match model_key:
            case key if "gpt" in key:
                interaction = OpenAIInteraction(model_config)
            case key if any (substring in key for substring in ["opus", "haiku", "sonnet"]):
                interaction = AnthropicInteraction(model_config)
            case key if any (substring in key for substring in ["malware-url", "malicious-url"]): 
                interaction = FineTunedInteraction(model_config)
            case key if "deepseek" in key:
                interaction = DeepSeekInteraction(model_config)
            case key if "llama" in key:
                interaction = LlamaInteraction(model_config)
            case _:
                raise ValueError(f"Unsupported model name: {model_config.model_name}")

        print(f"Returning interaction class of type: {type(interaction)}")
        return interaction