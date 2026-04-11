## Demo for project R5
### Repository for demo of prompting engineering techniques
This repository contains demonstrations and experiments for prompt engineering approaches using various language models. The notebooks illustrate different prompting strategies applied to language models for improved performance on specific tasks. Running Notebook files on Colab is recommended for better computational resources.

## Project Overview
This project explores prompt engineering techniques with state-of-the-art language models including BERT and FlanT5. The experiments demonstrate two main prompting approaches:
1. **Cloze Prompting**: Using masked language modeling where specific tokens are masked and predicted
2. **Prefix Prompting**: Using prefix-based sequences to guide model predictions

## Project Structure

### `anhhq/` - Main R5 Implementation
- **SML-R5.ipynb**: The main notebook containing the core R5 project implementation and experiments
  - Overview of the R5 project framework and methodology
  - Integration of prompt engineering techniques
  - Experimental results and analysis

### `NguyenNamHoang_20252758M/` - Detailed Prompt Engineering Experiments
Collection of detailed experiments comparing different prompting approaches:

#### Cloze Prompting with BERT
- **manual_cloze_promt_with_bert.ipynb**: Comprehensive notebook implementing BERT with manual cloze prompting
  - Model: BERT (Bidirectional Encoder Representations from Transformers)
  - Approach: Manual cloze-style prompts where tokens are masked for prediction
  - Evaluation metrics and performance analysis
  - **Output**: `manual_cloze_promt_with_bert_predicted_result.csv` - Results containing predictions and metrics

#### Prefix Prompting with FlanT5
- **manual_prefix_prompt_with_flanT5.ipynb**: Comprehensive notebook implementing FlanT5 with manual prefix prompting
  - Model: FlanT5 (Fine-tuned Language Model on T5 architecture)
  - Approach: Manual prefix-based prompts to guide model behavior
  - Task-specific fine-tuning and evaluation
  - **Output**: `manual_prefix_prompt_with_flanT5_predicted_result.csv` - Results containing predictions and comparative metrics

## Getting Started
1. Clone this repository
2. Open the notebooks in Google Colab or a Jupyter environment
3. Follow the instructions in each notebook to run experiments
4. Review the generated CSV files for detailed results and predictions

## Requirements
- Python 3.7+
- PyTorch or TensorFlow
- Transformers library (Hugging Face)
- Jupyter or Google Colab
