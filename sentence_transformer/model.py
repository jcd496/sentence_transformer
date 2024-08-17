import torch
import torch.nn as nn
import lightning as pl
from transformers import BertTokenizer, BertModel


class TransformerModel(pl.LightningModule):
    def __init__(self, model_name='bert-base-uncased', learning_rate=2e-5, num_classes=2, task='finetune'):
        super(TransformerModel, self).__init__()

        # attributes for task 1
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        print(self.model.config)

        # attributes for task 2
        self.classification_head = nn.Linear(self.model.config.hidden_size, num_classes)
        self.sentiment_head = nn.Linear(self.model.config.hidden_size, 1)

        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

        self.learning_rate = learning_rate
        
        self.update_task(task)

    def freeze_encoder(self):
        for p in self.model.parameters():
            p.requires_grad = False

    def update_task(self, task):
        self.task = task
        if self.task == 'finetune':
            print('setting task: finetune')
            self.training_step = self.finetune_training_step
        elif self.task == 'classification':
            print('setting task: classification')
            self.training_step = self.classification_training_step
            self.freeze_encoder()
        elif self.task == 'sentiment':
            print('setting task: sentiment')
            self.training_step = self.sentiment_training_step
            self.freeze_encoder()
        
    def forward_embeddings(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state.mean(dim=1)  # Mean pooling

    def forward_classification_head(self, embedding):
        return self.classification_head(embedding)

    def forward_sentiment_head(self, embedding):
        return self.sentiment_head(embedding).squeeze()

    def forward(self, input_ids, attention_mask):
        embedding = self.forward_embeddings(input_ids, attention_mask)
        classification_output = self.forward_classification_head(embedding)
        sentiment_output = self.forward_sentiment_head(embedding)
        return embedding, classification_output, sentiment_output

    def similarity_loss(self, embeddings1, embeddings2, labels):
        # Compute cosine similarity loss
        similarity = torch.nn.functional.cosine_similarity(embeddings1, embeddings2)
        loss = self.bce_loss(similarity, labels)
        return loss

    def classification_loss(self, predicted_classes, class_labels):
       return self.ce_loss(predicted_classes, class_labels)
    
    def sentiment_loss(self, predicted_sentiment, sentiment_label):
       return self.bce_loss(predicted_sentiment, sentiment_label)

    def unpack_batch(self, batch):

        if self.task == 'finetune':
            # Extract data from batch
            input_ids1 = batch['input_ids1']
            attention_mask1 = batch['attention_mask1']
            input_ids2 = batch['input_ids2']
            attention_mask2 = batch['attention_mask2']
            labels = batch['label']
            return input_ids1, attention_mask1, input_ids2, attention_mask2, labels

        elif (self.task == 'classification') or (self.task == 'sentiment'):
          input_ids = batch['input_ids']
          attention_mask = batch['attention_mask']
          labels = batch['label']
          return input_ids, attention_mask, labels

    def finetune_training_step(self, batch, batch_idx):
        input_ids1, attention_mask1, input_ids2, attention_mask2, labels = self.unpack_batch(batch)
        print('finetune step')
        # Compute embeddings
        embeddings1  = self.forward_embeddings(input_ids1, attention_mask1)
        embeddings2  = self.forward_embeddings(input_ids2, attention_mask2)
        
        similarity_loss = self.similarity_loss(embeddings1, embeddings2, labels)

        return similarity_loss
    
    def classification_training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = self.unpack_batch(batch)
        
        # Compute embeddings
        embeddings  = self.forward_embeddings(input_ids, attention_mask)

        # Compute classes
        classes = self.forward_classification_head(embeddings)

        loss = self.classification_loss(classes, labels)
        return loss

    def sentiment_training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = self.unpack_batch(batch)

        # Compute embeddings
        embeddings  = self.forward_embeddings(input_ids, attention_mask)

        # Compute sentiment
        sentiment = self.forward_sentiment_head(embeddings)

        loss = self.sentiment_loss(sentiment, labels)
        return loss
  
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

    def predict_sentence(self, sentence):
        encoding = self.tokenizer(sentence, max_length=128, padding='max_length', truncation=True, return_tensors='pt')

        # Convert to tensors and flatten
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask'] 
        
        embedding, classification, sentiment = self(input_ids, attention_mask)
        classification = classification.argmax(-1)
        sentiment = nn.functional.sigmoid(sentiment)
        
        return embedding, classification, sentiment
