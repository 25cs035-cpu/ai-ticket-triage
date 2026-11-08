# AI Customer Support Ticket Triage System

A machine-learning app that reads a customer support message and predicts its **category**, **priority** and **support department**, shows a **confidence score** for each prediction, and flags uncertain tickets for **human review**.

## Features
- Single ticket analysis (type a message, get category, department, priority and confidence)
- Adjustable human-review threshold (sidebar slider)
- Bulk CSV upload with downloadable predictions
- Dashboard: total tickets, high-priority count, tickets needing review, and distributions by category, department and priority

## Dataset
[Tobi-Bueck/customer-support-tickets](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets) on Hugging Face (file `dataset-tickets-multi-lang-4-20k.csv`): 20,000 support tickets in English and German. License: CC BY-NC 4.0 (non-commercial use).
The data file is not stored in this repository. To retrain, download it and save it as `data/tickets.csv`.

Columns used: `subject` + `body` (input text); `type` (category), `queue` (department) and `priority` (labels).
Columns `answer` and `tag_1`..`tag_8` are not used, to avoid data leakage.

## How it works
1. `preprocessing.py` combines subject and body, lowercases the text and removes placeholders such as `<tel_num>`.
2. `train.py` trains three separate models (category, department, priority). Each is a scikit-learn Pipeline of TF-IDF (1-2 word phrases) followed by Logistic Regression with balanced class weights, trained on 80% of the tickets and tested on the other 20%.
3. `predict.py` loads the models, predicts a label for each output and uses the highest class probability as the confidence score. If any confidence is below the threshold, the ticket is flagged for human review.
4. `app.py` is the Streamlit interface.

## Results (4,000 held-out test tickets)

| Model | Accuracy | Macro F1 |
|---|---|---|
| Category (type) | 0.797 | 0.81 |
| Department (queue) | 0.422 | 0.40 |
| Priority | 0.543 | 0.53 |

These are modest results. Category is reasonably good; department (10 classes) and priority are hard to predict from text alone. Confidence scores and the human-review flag exist because of this.

## Limitations
- The labels in the dataset look noisy in places, which may limit accuracy.
- Small departments (for example Returns and Exchanges, Sales and Pre-Sales) are rarely predicted correctly.
- The dataset has no "Critical" priority, only low, medium and high.
- Data is a mix of English and German; no language-specific handling is used.
- Dataset license is non-commercial.

## Run locally (Windows, PowerShell)
```
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```
To retrain the models: place the dataset at `data/tickets.csv`, then run `python train.py`.

## Project structure
```
app.py             Streamlit app
predict.py         load models and predict one ticket
preprocessing.py   text cleaning and data loading
train.py           train and save the three models
evaluate.py        confidence-threshold analysis
models/            trained models (.pkl)
requirements.txt   Python dependencies
```
