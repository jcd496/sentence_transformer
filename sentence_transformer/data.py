import torch
from torch.utils.data import Dataset, DataLoader


class SentencePairDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=128):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        sentence1 = example['sentence1']
        sentence2 = example['sentence2']
        label = example['label']

        # Tokenize sentences
        encoding1 = self.tokenizer(sentence1, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        encoding2 = self.tokenizer(sentence2, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')

        # Convert to tensors and flatten
        input_ids1 = encoding1['input_ids'].squeeze(0)
        attention_mask1 = encoding1['attention_mask'].squeeze(0)
        input_ids2 = encoding2['input_ids'].squeeze(0)
        attention_mask2 = encoding2['attention_mask'].squeeze(0)
        label = torch.tensor(label, dtype=torch.float)

        return {
            'input_ids1': input_ids1,
            'attention_mask1': attention_mask1,
            'input_ids2': input_ids2,
            'attention_mask2': attention_mask2,
            'label': label
        }


def prepare_finetune_data(tokenizer):
    # Example data (sentence pairs and labels)
    sentences1 = ["This is a sentence.", "Another sentence here."]
    sentences2 = ["This is a sentence.", "A different sentence."]
    labels = [1, 0]  # 1 if similar, 0 if dissimilar

    # Convert data to a format compatible with Dataset
    train_examples = [{'sentence1': s1, 'sentence2': s2, 'label': label} for s1, s2, label in zip(sentences1, sentences2, labels)]

    # Create dataset and dataloader
    train_dataset = SentencePairDataset(train_examples, tokenizer)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=2)

    return train_dataloader

class ClassificationDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=128, target_type=torch.long):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.target_type = target_type

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        sentence = example['sentence']
        label = example['label']

        # Tokenize sentence
        encoding = self.tokenizer(sentence, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')

        # Convert to tensors and flatten
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        label = torch.tensor(label, dtype=self.target_type)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': label
        }

def prepare_sentiment_data(tokenizer):
    # Example data (sentences and labels)
    sentences = ["This is a positive sentence.", "This is a negative sentence."]
    labels = [1, 0]  # 1 for positive, 0 for negative

    # Convert data to a format compatible with Dataset
    train_examples = [{'sentence': s, 'label': label} for s, label in zip(sentences, labels)]

    # Initialize tokenizer and create dataset and dataloader
    train_dataset = ClassificationDataset(train_examples, tokenizer, target_type=torch.float32)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=2)

    return train_dataloader

def prepare_classification_data(tokenizer):
    # Example data (sentences and labels for 4 classes)
    sentences = [
        "This is a positive sentence for class 0.",
        "This is a negative sentence for class 1.",
        "This sentence belongs to class 2.",
        "This is an example for class 3."
    ]
    labels = [0, 1, 2, 3]  # 4 different classes

    # Convert data to a format compatible with Dataset
    train_examples = [{'sentence': s, 'label': label} for s, label in zip(sentences, labels)]

    # Initialize tokenizer and create dataset and dataloader
    train_dataset = ClassificationDataset(train_examples, tokenizer)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=2)

    return train_dataloader

