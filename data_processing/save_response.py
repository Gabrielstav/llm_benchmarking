from threading import Lock 
import pandas as pd 
from pathlib import Path 
import json
from configs.config_loader import ModelConfig
from containers.data_container import ModelResponse

# TODO: 
# Add support for writing to JSON - make configurable 
# Add support for writing as JSON (and CSV?) to MongoDB (copy structure and logic from AI API cache)
#   Make new database in new collection for the task (LLM benchmark) 
#   Test writing/deleting etc in mock DB.

class SaveResponse: 
    """
    Class to handle saving model responses to CSV files.

    Attributes:
        model_name (str): The name of the model, used to name the CSV file.
        buffer_size (int): The size of the buffer before flushing to the CSV file.
        config (ModelConfig): The configuration for the model.
    """

    def __init__(self, model_name: str, buffer_size: int, config: ModelConfig):
        self.config = config 
        self.buffer = []
        self.model_name = model_name.replace("/", "-") 
        self.buffer_size = buffer_size
        self.lock = Lock()
        self.file_path = self._resolve_csv_path()
        self._ensure_csv_initialized()
        self.metadata_path = self._resolve_metadata_path()

    def add_data(self, model_response):
        """
        Adds data to the buffer and flushes the buffer to the CSV file if the buffer size is exceeded.

        Args:
            model_response (ModelResponse): The model response containing analysis results to be added to the buffer.
        """
        should_flush = False
        with self.lock:
            print(f"Adding data from dataclass to buffer")
            for domain_analysis in model_response.analysis_results:
                self.buffer.append({
                    "domain_name": domain_analysis.domain_name,
                    "category": domain_analysis.category,
                    "confidence_score": domain_analysis.confidence_score,
                    "reasons": str(domain_analysis.reasons)
                })
            print(f"Buffer size after adding data: {len(self.buffer)}")
            if len(self.buffer) >= self.buffer_size:
                should_flush = True  # set flag to flush

        # flushing buffer (outside of lock)
        if should_flush:
            self._flush_buffer()

        # update the ModelResponse instance with output_data_path
        model_response.output_data_path = self.file_path
        # save metadata 
        self._save_metadata(model_response)

    def _flush_buffer(self):
        """
        Flushes the buffer to the CSV file.
        """
        with self.lock:
            if self.buffer:
                new_df = pd.DataFrame(self.buffer)
                if self.file_path.exists():
                    existing_df = pd.read_csv(self.file_path, usecols=["domain_name"])
                    # skip adding new domains if they are already present 
                    new_df = new_df[~new_df["domain_name"].isin(existing_df["domain_name"])]
                if not new_df.empty:
                    new_df.to_csv(self.file_path, mode="a", header=False, index=False)
                self.buffer = []
                print(f"Buffer flushed and cleared") 

    def _resolve_csv_path(self):
        """
        Resolves the path to the CSV file for the model responses.

        Returns:
            Path: The resolved path to the CSV file.
        """
        # Navigate up to the llm_benchmark directory
        current_dir = Path(__file__).resolve().parent
        llm_benchmark_dir = current_dir.parent
        model_response_dir = llm_benchmark_dir / "output_data"

        if self.config.single_mode:
            single_mode_dir = model_response_dir / "single_processing"
            model_response_csv_dir = single_mode_dir / "model_responses_csv"
        else: 
            batch_mode_dir = model_response_dir / "batch_processing"
            model_response_csv_dir = batch_mode_dir / "model_responses_csv"

        model_response_csv_dir.mkdir(parents=True, exist_ok=True)
        model_response_file = model_response_csv_dir / f"{self.model_name}.csv"
        
        if not model_response_file.exists():
            model_response_file.touch()

        print(f"CSV file path resolved to: {model_response_file}")
        return model_response_file
    
    def _resolve_metadata_path(self) -> Path:
        """
        Resolves the path to the metadata file for the model responses.

        Returns:
            Path: The resolved path to the metadata file.
        """
        # Navigate up to the llm_benchmark directory
        current_dir = Path(__file__).resolve().parent
        llm_benchmark_dir = current_dir.parent  

        # Set directory based on single_mode or batch_mode
        model_response_dir = llm_benchmark_dir / "output_data"
        if self.config.single_mode:
            mode_dir = model_response_dir / "single_processing"
        else:
            mode_dir = model_response_dir / "batch_processing"

        metadata_dir = mode_dir / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        metadata_file = metadata_dir / f"{self.model_name}.json"

        return metadata_file
    
    def _ensure_csv_initialized(self):
        """
        Ensures that the CSV file is initialized with headers if it does not exist or is empty.
        """
        # if file doesn't exist or is empty, initialize headers
        if not self.file_path.exists() or self.file_path.stat().st_size == 0:
            df = pd.DataFrame(columns=["domain_name", "category", "confidence_score", "reasons"])
            df.to_csv(self.file_path, index=False)
            print("CSV initialized")

    def _save_metadata(self, model_response: ModelResponse):
        """
        Save metadata for the model response to a JSON file.

        Args:
            model_response (ModelResponse): The model response containing metadata to be saved.
        """
        if not self.metadata_path.exists():
            metadata = {
                "model_name": model_response.model_name,
                "output_data_path": str(model_response.output_data_path),
                "metadata": model_response.metadata.dict() if model_response.metadata else {}
            }
            with open(self.metadata_path, "w") as f:
                json.dump(metadata, f)
                
            print(f"Metadata saved to: {self.metadata_path}")