from transformers import pipeline
import streamlit as st
import json
import numpy as np
import logging
import re

logging.basicConfig(filename='article.log', encoding='utf-8', level=logging.INFO)


class ArticleClassifyError(Exception):
    pass


@st.cache_data
def load_data():
    with open('arxivData.json') as file:
        data = json.load(file)
    return data


@st.cache_resource
def load_model():
    return pipeline('zero-shot-classification', model='facebook/bart-large-mnli')


def has_cyrillic(text):
    return bool(re.search('[а-яА-Я]', text))


def validate_request(text: str):
    if len(text) < 20:
        logging.warning(f'text is not valid. Text: {text}, error: Too short text (minimum 20 characters)')
        raise ArticleClassifyError('Too short text (minimum 20 characters)')
    if has_cyrillic(text):
        logging.warning(f'text is not valid. Text: {text}, error: Only English is allowed')
        raise ArticleClassifyError('Only English is allowed')


date = load_data()
classifier = load_model()

'''
# creating dataset
dataset = load_dataset('json', data_files='arxivData.json')
dataset['train'].remove_columns(['author', 'day', 'id', 'link', 'month', 'title', 'year'])
texts = [article['summary'] for article in data]
sep = 40000
train_texts = texts[:sep]
test_texts = texts[sep:]
labels = [classes.index(eval(article['tag'])[0]['term'].split('.')[0]) for article in data]
train_labels = labels[:sep]
test_labels = labels[sep:]
train = [[train_texts[ind], train_labels[ind]] for ind in range(len(train_texts))]
test = [[test_texts[ind], test_labels[ind]] for ind in range(len(test_texts))]
from datasets import Dataset, Features, Value, ClassLabel, DatasetDict
train_ds = Dataset.from_dict({"text": train_texts, 'label': train_labels}, features=Features({"text":  Value(dtype='string', id=None), "label": ClassLabel(num_classes=len(classes), names=classes, names_file=None, id=None)}))
test_ds = Dataset.from_dict({"text": test_texts, 'label': test_labels}, features=Features({"text":  Value(dtype='string', id=None), "label": ClassLabel(num_classes=len(classes), names=classes, names_file=None, id=None)}))
ds = DatasetDict({'train': train_ds, 'test': test_ds})

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)
tokenized_ds = ds.map(preprocess_function, batched=True)

from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

import evaluate
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)
    
id2label = {0: 'adap-org', 1: 'astro-ph', 2: 'cmp-lg', 3: 'cond-mat', 4: 'cs', 5: 'econ', 6: 'eess',
            7: 'gr-qc', 8: 'hep-ex', 9: 'hep-lat', 10: 'hep-ph', 11: 'hep-th', 12: 'math', 13: 'nlin',
            14: 'nucl-th', 15: 'physics', 16: 'q-bio', 17: 'q-fin', 18: 'quant-ph', 19: 'stat'}
label2id = {'adap-org': 0, 'astro-ph': 1, 'cmp-lg': 2, 'cond-mat': 3, 'cs': 4, 'econ': 5, 'eess': 6,
            'gr-qc': 7, 'hep-ex': 8, 'hep-lat': 9, 'hep-ph': 10, 'hep-th': 11, 'math': 12, 'nlin': 13,
            'nucl-th': 14, 'physics': 15, 'q-bio': 16, 'q-fin': 17, 'quant-ph': 18, 'stat': 19}
            
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=20, id2label=id2label, label2id=label2id
)

training_args = TrainingArguments(
    output_dir="my_model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
'''

st.markdown("### Welcome!\n I can classify your article by topic:")
st.markdown("mathematics, biology, history, machine learning, neural networks, programming, literature, medicine")
st.markdown("<img width=200px src='https://shapka-youtube.ru/wp-content/uploads/2022/07/avatarka-kot-uchenyy.jpg'>",
            unsafe_allow_html=True)
title = st.text_area("Article title:")
abstract = st.text_area("Article abstract:")
labels = ["mathematics", "biology", "history", "machine learning", "neural networks", "programming", "literature",
          "medicine"]
try:
    if title:
        text = title
        if abstract:
            text += abstract
        validate_request(text)
        prediction = classifier(text, labels)
        cumsum = np.cumsum(prediction['scores'])
        top_limit = next(index for index, value in enumerate(cumsum) if value >= 0.95)
        st.markdown('Top 95% topics:')
        for i in range(top_limit + 1):
            st.markdown(prediction['labels'][i] + ': ' + str(int(np.round(prediction['scores'][i], 2) * 100)) + '%')
    else:
        st.markdown('Article title is not entered')
except ArticleClassifyError as error:
    st.markdown(str(error))
