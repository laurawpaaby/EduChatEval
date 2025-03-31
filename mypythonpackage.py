### 1. THIS FIRST BIT IS FOR THE FIRST STEP OF THE PIPELINE: FRAMEWORK GENERATION
### THIS CREATES THE SYNTHETIC DATASET USED FOR TRAINING THE CLASSIFIER AND IS THE FOUNDATION FOR THE ANALYSIS
      
from src.framework_generation.outline_synth_LMSRIPT import SyntheticDataGenerator
from src.framework_generation.train_tinylabel_classifier import QualityFilterPipeline

class FrameworkGenerator:
    """
    Orchestrates the full framework generation pipeline using synthetic data generation and quality filtering.

    Arguments:
    - prompt_path (str, required): Path to the Python file containing the prompt dictionary.
    - save_json_path (str, required): Path to save the generated JSON file.
    - save_csv_path (str, required): Path to save the generated CSV file.
    - model_name (str, optional): Name of the model to use for text generation.
    - num_samples (int, optional): Number of samples to generate for each category.
    - csv_path (str, required): Path to the CSV file containing true data for quality filtering.
    - text_column (str, required): Name of the column containing text data in the CSV file.
    - label_column (str, required): Name of the column containing labels in the CSV file.
    - split_ratio (float, optional): Ratio for train-test split.
    - training_params (list, optional): Parameters for the training process.
    - tuning (bool, optional): Whether to perform hyperparameter tuning.
    - tuning_params (dict, optional): Parameters for hyperparameter tuning.
    - save_model_path (str, required): Path to save the trained model.
    - filtered_save_path (str, required): Path to save the filtered dataset.
    """


    # for each class we have an innit function that takes the parameters for the pipeline - maybe too many inputs??????
    def __init__(
        self,
        prompt_path="src/framework_generation/outline_prompts/prompt_default.py",
        save_json_path="data/generated_tuning_data/generated_data.json",
        save_csv_path="data/generated_tuning_data/generated_data.csv",
        model_name="llama-3.2-3b-instruct",
        num_samples=500,
        csv_path="data/tiny_labeled_data.csv",
        text_column="text",
        label_column="category",
        split_ratio=0.2,
        training_params=[0.01, 'cross_entropy', 5e-5, 8, 8, 4, 0.01],
        tuning=False,
        tuning_params=None,
        save_model_path="Models/package_quality_classifier",
        filtered_save_path="data/generated_tuning_data/final_filtered.csv"
    ):
        self.prompt_path = prompt_path
        self.save_json_path = save_json_path
        self.save_csv_path = save_csv_path
        self.model_name = model_name
        self.num_samples = num_samples

        self.csv_path = csv_path
        self.text_column = text_column
        self.label_column = label_column
        self.split_ratio = split_ratio
        self.training_params = training_params
        self.tuning = tuning
        self.tuning_params = tuning_params
        self.save_model_path = save_model_path
        self.filtered_save_path = filtered_save_path

    def run(self):
        """
        Runs the full pipeline for framework generation.
        """

        # Step 1: Generate synthetic data from prompts
        synth_generator = SyntheticDataGenerator(
            prompt_path=self.prompt_path,
            save_json_path=self.save_json_path,
            save_csv_path=self.save_csv_path,
            model=self.model_name,
            samples=self.num_samples
        )
        synth_generator.run()

        # Step 2: Train quality model on true data and use it to filter the synthetic dataset
        filter_pipeline = QualityFilterPipeline(
            csv_path=self.csv_path,
            model_name=self.model_name,
            text_column=self.text_column,
            label_column=self.label_column,
            split_ratio=self.split_ratio,
            training_params=self.training_params,
            tuning=self.tuning,
            tuning_params=self.tuning_params,
            save_path=self.save_model_path,
            synth_path=self.save_csv_path,
            filtered_save_path=self.filtered_save_path
        )
        filter_pipeline.run()
        print("Framework generation complete.")


#### 2. NOW NEXT STEP SHOULD BE GENERATING THE SYNTHETIC DIALOGUE DATA ..........
class DialogueGenerator:
    """
    Generates synthetic dialogue data using a trained quality classifier.
    """

    #def __init__(
    #) 