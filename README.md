# Fine-Tuning for Sequence Classification

This repository contains code for fine-tuning a pre-trained BERT model for sequence classification using the Yelp review dataset.

## Prerequisites

Make sure you have the following libraries installed:

- `datasets`
- `transformers`
- `evaluate`
- `accelerate`

You can install these libraries using pip:

```shell
pip install datasets transformers transformers[torch] evaluate accelerate
```

## Usage

1. Load the Yelp review dataset:

```python
from datasets import load_dataset

dataset = load_dataset("yelp_review_full")
```

2. Tokenize the dataset using the BERT tokenizer:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def tokenize_functions(inputs):
    return tokenizer(inputs["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_functions, batched=True)

train_dataset = tokenized_datasets["train"]
eval_dataset = tokenized_datasets["test"]
```

3. Prepare a smaller subset of the dataset for training and evaluation:

```python
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
```

4. Load the pre-trained BERT model for sequence classification:

```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)
```

5. Define evaluation metrics:

```python
import numpy as np
import evaluate

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
```

6. Initialize the Accelerator for faster training:

```python
from accelerate import Accelerator

accelerator = Accelerator()
```

7. Configure the training arguments:

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="test_trainer",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="epoch",
)
```

8. Create a Trainer object and start training:

```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
    accelerator=accelerator,
)

trainer.train()
```

## Customization

- If you want to use a different pre-trained BERT model, modify the `from_pretrained` method in the `AutoModelForSequenceClassification.from_pretrained` line with the desired model name.

- Adjust the training arguments (`TrainingArguments`) based on your specific requirements. You can change the number of training epochs, batch sizes, evaluation strategies, logging strategies, and save strategies according to your needs.

## References

- [Original Colab Notebook](https://colab.research.google.com/drive/1pMod_5BWVZ1EGWciZDIO8hHmv_PBjHSG)
