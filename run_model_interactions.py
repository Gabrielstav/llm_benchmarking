import asyncio
from typing import List 
from configs.config_loader import ModelConfig, create_model_configs
from managers.interaction_manager import ModelInteractionManager
from managers.benchmark_manager import run_analysis, run_highlighted_comparison


# General TODO: 
# Make prompt comparison code that automatically scores prompts against validated data, select prompts and run the manager 
# Add markdown documentation for adding a new model (model config, subclass, model key)
# Use Instructor for all ModelInteraction subclasses to process model responses? see: https://github.com/jxnl/instructor
# Add delay between calls (because API limit is exceeded for anthropic models so easily)

async def run_model_async(configurations):
    manager = ModelInteractionManager(configurations=configurations, buffer_size=25)
    print(f"Configs in manager call: {manager.configurations}")
    await manager.run_interactions()

def async_run(configurations):
    print(f"Configs in async run: {configurations}")
    asyncio.run(run_model_async(configurations=configurations))

def run_models(selected_models: List[str], batch_size: int, total_domains: int, single_mode: bool, start_index: int = 0):

    if "all" in selected_models:
        selected_models = ModelConfig.list_all_model_names()
    
    model_configs = create_model_configs(
        model_names=selected_models,
        batch_size=batch_size,
        total_domains=total_domains,
        single_mode=single_mode,
        start_index=start_index
    )
    
    print(f"Configurations in run model: {model_configs}")

    async_run(configurations=model_configs)


if __name__ == "__main__":
    """
    # EXAMPLES:

    # First we make API calls and store the responses:
    # Get LLM responses from all supported models (not all models can run in batch_mode or with batch_size > 1):
    run_models(selected_models=["all"], batch_size=10, total_domains=300, single_mode=False, start_index=0)
    # Or select specific models:
    run_models(selected_models=["gpt4", "opus"], batch_size=20, total_domains=40, single_mode=True, start_index=0)

    # Then we can benchmark the models using the saved responses: 
    run_analysis(selected_models=["gpt4-o", "malware-url-detect", "llama", "gpt-4-1106-preview"], mode="single", analyses=["false_positives"], 
    save_plots=False, numerical_output=False)
    # Can also just benchmark all available models and use all statistics: 
    run_analysis(selected_models=["gpt4-o", "llama"], mode="single", analyses=["all"], save_plots=False, numerical_output=True)

    # Lastly, make highlighted comparison (the colored .xlsx file comparing model responses to validated data):
    run_highlighted_comparison(["gpt4-o", "llama"], mode="batch")
    """

    run_models(selected_models=["gpt4-o", "llama"], batch_size=1, total_domains=2, single_mode=True, start_index=0)
    # run_analysis(selected_models=["gpt4-o", "llama"], mode="single", analyses=["all"], save_plots=False, numerical_output=False)
    # run_highlighted_comparison(selected_models=["gpt4-o", "llama"], mode="single")







                   

