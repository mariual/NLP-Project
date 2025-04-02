from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer,GPT2Tokenizer,RobertaTokenizer,ElectraTokenizer
from transformers import AutoTokenizer
from datasets import load_dataset
import pandas as pd
import torch

def get_emotion_dataset():
    return load_dataset('dair-ai/emotion')

def create_validation_split(df, test_size=0.2):
    # Splitting the data into training and validation sets (80% train, 20% validation)
    X_train, X_val, y_train, y_val = train_test_split(df['text'], df['label'], test_size=test_size, random_state=42)
    return X_train, X_val, y_train, y_val


def naive_bayes_preprocessing(remove_stopwords=False, use_bigrams=False):
    
    dataset = load_dataset('dalopeza98/isear-cleaned-dataset')
    
    vectorizer = CountVectorizer(
        stop_words='english' if remove_stopwords else None,  # remove stopwords
        ngram_range=(1, 2) if use_bigrams else (1, 1)        # use bigrams
    )

    processed_data = {}
    for split, data in dataset.items():
        df = pd.DataFrame(data)

        # Split the train set into training and validation sets
        if split == 'train':
            X_train, X_val, y_train, y_val = create_validation_split(df)
            # Transform the data
            X_train = vectorizer.fit_transform(X_train)
            X_val = vectorizer.transform(X_val)
            processed_data['train'] = (X_train, y_train)
            processed_data['validation'] = (X_val, y_val)
        else:
            X = vectorizer.transform(df['text'])  # Transform the test set
            y = df['label'].values
            processed_data[split] = (X, y)

    return processed_data, vectorizer

def goemotions_preprocessing(model_name="justin871030/bert-base-uncased-goemotions-original-finetuned", max_length=50):
    dataset = load_dataset("google-research-datasets/go_emotions", "simplified")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def one_hot(labels, num_classes=28):
        vec = torch.zeros(num_classes)
        vec[labels] = 1
        return vec

    processed_data = {}
    for split in ['train', 'validation', 'test']:
        ds = dataset[split]
        texts = ds['text']
        labels = ds['labels']

        tokenized = tokenizer(texts, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
        one_hot_labels = torch.stack([one_hot(lbl) for lbl in labels])

        processed_data[split] = (
            tokenized['input_ids'],
            tokenized['attention_mask'],
            one_hot_labels
        )

    return processed_data, tokenizer

def bert_preprocessing():


    dataset = load_dataset('dalopeza98/isear-cleaned-dataset')
    tokenizer = BertTokenizer.from_pretrained('bhadresh-savani/bert-base-uncased-emotion')

    processed_data = {}
    for split, data in dataset.items():
        df = pd.DataFrame(data)

        if split == 'train':
            # Split the train data into train and validation
            X_train, X_val, y_train, y_val = create_validation_split(df)
            
            # Tokenize the training data
            tokenized_train = tokenizer(X_train.tolist(), padding='max_length', truncation=True, max_length=70, return_tensors='pt')
            tokenized_val = tokenizer(X_val.tolist(), padding='max_length', truncation=True, max_length=70, return_tensors='pt')
            
            processed_data['train'] = (tokenized_train['input_ids'], tokenized_train['attention_mask'], torch.tensor(y_train.values))
            processed_data['validation'] = (tokenized_val['input_ids'], tokenized_val['attention_mask'], torch.tensor(y_val.values))
        else:
            # Tokenize the test data
            tokenized_test = tokenizer(df['text'].tolist(), padding='max_length', truncation=True, max_length=70, return_tensors='pt')
            processed_data[split] = (tokenized_test['input_ids'], tokenized_test['attention_mask'], torch.tensor(df['label'].values))

    return processed_data, tokenizer


def gpt2_preprocessing():
    dataset = load_dataset('dalopeza98/isear-cleaned-dataset')
    tokenizer = GPT2Tokenizer.from_pretrained('heegyu/gpt2-emotion')
    
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    processed_data = {}
    for split, data in dataset.items():
        df = pd.DataFrame(data)

        if split == 'train':
            # Split the train data into train and validation
            X_train, X_val, y_train, y_val = create_validation_split(df)
            
            # Tokenize the training data
            tokenized_train = tokenizer(X_train.tolist(), padding='max_length', truncation=True, max_length=70, return_tensors='pt')
            tokenized_val = tokenizer(X_val.tolist(), padding='max_length', truncation=True, max_length=70, return_tensors='pt')
            
            processed_data['train'] = (tokenized_train['input_ids'], tokenized_train['attention_mask'], torch.tensor(y_train.values))
            processed_data['validation'] = (tokenized_val['input_ids'], tokenized_val['attention_mask'], torch.tensor(y_val.values))
        else:
            # Tokenize the test data
            tokenized_test = tokenizer(df['text'].tolist(), padding='max_length', truncation=True, max_length=70, return_tensors='pt')
            processed_data[split] = (tokenized_test['input_ids'], tokenized_test['attention_mask'], torch.tensor(df['label'].values))

    return processed_data, tokenizer


def roberta_preprocessing():
    dataset = load_dataset('dalopeza98/isear-cleaned-dataset')
    tokenizer = RobertaTokenizer.from_pretrained('Dimi-G/roberta-base-emotion')

    processed_data = {}
    for split, data in dataset.items():
        df = pd.DataFrame(data)

        if split == 'train':
            # Split the train data into train and validation
            X_train, X_val, y_train, y_val = create_validation_split(df)
            
            # Tokenize the training data
            tokenized_train = tokenizer(X_train.tolist(), padding='max_length', truncation=True, max_length=70, return_tensors='pt')
            tokenized_val = tokenizer(X_val.tolist(), padding='max_length', truncation=True, max_length=70, return_tensors='pt')
            
            processed_data['train'] = (tokenized_train['input_ids'], tokenized_train['attention_mask'], torch.tensor(y_train.values))
            processed_data['validation'] = (tokenized_val['input_ids'], tokenized_val['attention_mask'], torch.tensor(y_val.values))
        else:
            # Tokenize the test data
            tokenized_test = tokenizer(df['text'].tolist(), padding='max_length', truncation=True, max_length=70, return_tensors='pt')
            processed_data[split] = (tokenized_test['input_ids'], tokenized_test['attention_mask'], torch.tensor(df['label'].values))

    return processed_data, tokenizer


def electra_preprocessing():
    dataset = load_dataset('dalopeza98/isear-cleaned-dataset')
    tokenizer = ElectraTokenizer.from_pretrained('mudogruer/electra-emotion')

    processed_data = {}
    for split, data in dataset.items():
        df = pd.DataFrame(data)

        if split == 'train':
            # Split the train data into train and validation
            X_train, X_val, y_train, y_val = create_validation_split(df)
            
            # Tokenize the training data
            tokenized_train = tokenizer(X_train.tolist(), padding='max_length', truncation=True, max_length=70, return_tensors='pt')
            tokenized_val = tokenizer(X_val.tolist(), padding='max_length', truncation=True, max_length=70, return_tensors='pt')
            
            processed_data['train'] = (tokenized_train['input_ids'], tokenized_train['attention_mask'], torch.tensor(y_train.values))
            processed_data['validation'] = (tokenized_val['input_ids'], tokenized_val['attention_mask'], torch.tensor(y_val.values))
        else:
            # Tokenize the test data
            tokenized_test = tokenizer(df['text'].tolist(), padding='max_length', truncation=True, max_length=70, return_tensors='pt')
            processed_data[split] = (tokenized_test['input_ids'], tokenized_test['attention_mask'], torch.tensor(df['label'].values))

    return processed_data, tokenizer