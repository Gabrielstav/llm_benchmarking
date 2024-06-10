from dataclasses import dataclass
from typing import Optional, Dict, List 
from pathlib import Path
import os 
import json
from dotenv import load_dotenv



# TODO:
# Add validation of data in ModelConfig, check for total_domains >= batch_size, , etc 

# path to same validated data but in the script dir:  
# def get_validated_data() -> Path:
#     return Path(__file__).resolve().parents[2] / "scripts" / "domains_data" / "testing_set" / "20240117-domain_risk_score_training_dataset.csv"


def get_validated_data() -> Path: 
    """
    Return: 
        Path to validated data in model_benchmarking/validated_data
    """
    return Path(__file__).resolve().parents[1] / "validated_data" / "318_domains.csv"

def get_model_config_file() -> Path: 
    """
    Return: 
        Path to static model configs in model_benchmarking/model_configs.json  
    """
    return Path(__file__).resolve().parents[0] / "model_configs.json"

def get_output_data_directory() -> Path: 
    """
    Return: 
        Path to output data dir for results, model_benchmarking/output_data
    """
    return Path(__file__).resolve().parents[1] / "output_data" 

def get_model_name_mapping() -> Dict[str, str]:
    """
    Return: 
        model_mapping (Dict[str, str]): Mapping for short model names to actual model names. 
    """
    with open(get_model_config_file(), "r") as f:
        config = json.load(f)
    
    model_mapping = {short_name: details["model_name"] for short_name, details in config["models"].items()}
    return model_mapping

def get_env_file() -> Path: 
    """
    Return: 
        Path to .env file in model_benchmarking dir (not the .env file used in iQ AI API)
    """
    return Path(__file__).resolve().parents[1] / ".env"



@dataclass
class ModelConfig:
    """
    Dataclass to hold the configuration for a model.

    Attributes:
        model_name (str): The name of the model.
        batch_size (int): The size of the batches to process.
        start_index (int): The starting index for processing data.
        total_domains (int): The total number of domains to process.
        single_mode (bool): Whether the model runs in single mode or batch mode.
        max_tokens (int): The maximum number of tokens for the model's output.
        api_key (str): The API key for accessing the model.
        end_point (str): The endpoint URL for the model.
        temperature (Optional[float]): The temperature setting for the model.
        organization_id (Optional[str]): The organization ID for the model (if applicable).
    """
        
    model_name: str 
    batch_size: int  
    total_domains: int 
    start_index: int   
    single_mode: bool  
    max_tokens: int
    api_key: str
    end_point: str
    temperature: Optional[float] = None 
    organization_id: Optional[str] = None

    @staticmethod
    def load_config(model_key: str) -> Dict:
        """
        Loads the configuration for a specific model from the JSON config file.

        Args:
            model_key (str): The key for the model configuration to load.
        Returns:
            Dict: The configuration settings for the specified model.
        Raises:
            ValueError: If the configuration for the specified model_key is not found.
        """
        # load the .env file specific to model_benchmarking 
        env_path = get_env_file()
        load_dotenv(dotenv_path=env_path)
        
        # load the configuration file
        config_path = get_model_config_file()
        with open(config_path, "r") as file:
            config_data = json.load(file)["models"].get(model_key)
            if not config_data:
                raise ValueError(f"Configuration for {model_key} not found")
            
            # load environment variables from .env in model_benchmarking
            config_data["api_key"] = os.getenv(config_data["api_key"])
            if "gpt" in model_key: 
                config_data["organization_id"] = os.getenv(config_data.get("organization_id"))
            
            return config_data

    @classmethod
    def create_config(cls, model_name: str, **runtime_settings):
        """
        Creates a ModelConfig instance by merging static settings from the config file with runtime settings.

        Args:
            model_name (str): The name of the model.
            **runtime_settings: Additional runtime settings to override static settings.
        Returns:
            ModelConfig: An instance of the ModelConfig class with merged settings.
        """
        static_settings = cls.load_config(model_name)
        # use the model name from the config file 
        full_model_name = static_settings["model_name"]
        # remove duplicate model name from static settings
        static_settings.pop("model_name", None)  
        return cls(model_name=full_model_name, **static_settings, **runtime_settings)
    
    @staticmethod
    def list_all_model_names() -> List[str]:
        """
        Lists all model names from the JSON config file.

        Returns:
            List[str]: A list of all model keys (shorthand names).
        """
        config_path = get_model_config_file()
        with open(config_path, "r") as file:
            model_keys = json.load(file)["models"].keys()
        return list(model_keys)
    
def create_model_configs(model_names: List[str], batch_size: int, total_domains: int, single_mode: bool, start_index: int = 0) -> List[ModelConfig]:
    model_configs = []
    for model_name in model_names:
        config = ModelConfig.create_config(
            model_name=model_name,
            batch_size=batch_size,
            total_domains=total_domains,
            single_mode=single_mode,
            start_index=start_index
        )
        model_configs.append(config)

    print(f"Model config created: {config}")
    return model_configs