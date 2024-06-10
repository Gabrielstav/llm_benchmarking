import pandas as pd 
from model_benchmarking.configs.prompts import get_domain_prompt_single, get_domain_prompt_batch, get_only_domain_single
from model_benchmarking.configs.config_loader import ModelConfig

class ConstructPrompt:
    """
    Class to construct prompts for model interactions based on the configuration and domain data.

    Attributes:
        domains_data (pd.DataFrame): The DataFrame containing domain data.
        config (ModelConfig): The configuration settings for the model.
    """

    def __init__(self, domains_data: pd.DataFrame, config: ModelConfig):
        self.domains_data = domains_data
        self.config = config 

    def construct_prompt(self) -> str:
        """
        Constructs the prompt for the model based on the configuration and domain data.
        Note: The structure of the validated data needs to have the two columns: Domains | Category. 
        If the columns are not named domains (containing the domains to test) and category (malicious or benign)
        the prompts cannot be constructed and the validation of the response against the validated data cannot be done. 

        Returns:
            str: The constructed prompt.
        Raises:
            ValueError: If no prompt template is found for the model configuration.
        """

        # return correct prompt template depending on config.single_mode and config.max_tokens
        match self.config:
            case config if config.single_mode and self.config.max_tokens > 513:
                template_str = get_domain_prompt_single()["template"]
            case config if not config.single_mode and self.config.max_tokens > 513:
                template_str = get_domain_prompt_batch()["template"]
            # just basing the prompt returned off of max token length currently,
            # so smaller models use the small prompts:  
            case config if self.config.max_tokens == 512:
                template_str = get_only_domain_single()["template"]
            case _: 
                raise ValueError(f"Prompt template not found for model config {self.config.model_name}. Make sure the model config in the model_configs.json file is correctly set up.")
        domain_list = ", ".join(self.domains_data["fqdn"])
        
        # insert domains in single prompt 
        if self.config.single_mode: 
            formatted_prompt = template_str.replace("{{domain_name}}", domain_list.strip())
        
        # insert domains in batch prompt
        else: 
            formatted_prompt = template_str.replace("{{num_of_items}}", str(self.config.batch_size))
            formatted_prompt = formatted_prompt.replace("{{domain_names}}", domain_list.strip())

        return formatted_prompt
