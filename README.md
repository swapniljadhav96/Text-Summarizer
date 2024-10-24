# Text Summarizer - Pegasus Model for Dialogue Summarization

## Overview

This project employs the Pegasus model for dialogue summarization on the SAMSum dataset, focusing on abstractive summarization tasks. The code provides a training pipeline, evaluation metrics, and an inference pipeline for generating summaries from new dialogues.

## Key Features

- **Pegasus Model:** Utilizes the Pegasus model, renowned for abstractive summarization tasks.
  
- **SAMSum Dataset:** The project relies on the SAMSum dataset, tailored for dialogue summarization, capturing natural language interactions.

- **Training Pipeline:** Implements a training pipeline for fine-tuning the Pegasus model on dialogue-summary pairs.

- **Evaluation Metrics:** Employs standard metrics like ROUGE for evaluating the quality of generated summaries.

- **Inference Pipeline:** Offers an inference pipeline to generate summaries from new dialogues using the trained Pegasus model.

## Usage

### Inference

Use the provided script to demonstrate the model's inference capabilities:

```python
# Prediction
gen_kwargs = {"length_penalty": 0.8, "num_beams": 8, "max_length": 128}

sample_text = dataset_samsum["test"][0]["dialogue"]
reference = dataset_samsum["test"][0]["summary"]

pipe = pipeline("summarization", model="pegasus-samsum-model", tokenizer=tokenizer)

# Print Results
print("Dialogue:")
print(sample_text)

print("\nReference Summary:")
print(reference)

print("\nModel Summary:")
print(pipe(sample_text, **gen_kwargs)[0]["summary_text"])
```

## Results

The project demonstrates state-of-the-art results on the SAMSum dataset, showcasing the effectiveness of the Pegasus model in generating informative and concise dialogue summaries.
