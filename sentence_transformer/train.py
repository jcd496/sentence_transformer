from model import TransformerModel
from data import prepare_finetune_data, prepare_classification_data, prepare_sentiment_data
import lightning as pl
import lovely_tensors

def make_trainer():
    # Callbacks
    checkpoint_callback = pl.pytorch.callbacks.ModelCheckpoint(every_n_train_steps=100,
                                                              save_top_k=3,
                                                              monitor='val_loss',
                                                              save_last=True)

    lr_monitor_callback = pl.pytorch.callbacks.LearningRateMonitor(
        logging_interval='step')

    # Initialize trainer
    trainer = pl.Trainer(
	accelerator='auto',
   	max_epochs=5,
	gradient_clip_val=1.0,
	accumulate_grad_batches=4,
        callbacks=[checkpoint_callback, lr_monitor_callback],
        fast_dev_run=True,
    )

    return trainer


def main():
    # Initialize model
    model_name = 'bert-base-uncased'
    model = TransformerModel(model_name=model_name, num_classes=4, task='finetune')

    # Prepare data
    finetune_train_dataloader = prepare_finetune_data(model.tokenizer)

    # Finetune train the model
    trainer = make_trainer()
    trainer.fit(model, finetune_train_dataloader)

    # Train classification head
    trainer = make_trainer()
    classification_train_dataloader = prepare_classification_data(model.tokenizer)
    
    model.update_task('classification')

    trainer.fit(model, classification_train_dataloader)

    # Train sentiment head
    trainer = make_trainer()
    sentiment_train_dataloader = prepare_sentiment_data(model.tokenizer)
    
    model.update_task('sentiment')

    trainer.fit(model, sentiment_train_dataloader)

    # Infer sentence embeddings with the fine-tuned model
    sentences = ["This is a new sentence.", "Yet another sentence."]
    encodings = model.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    embeddings = model.model(**encodings).last_hidden_state.mean(dim=1)

    # Print embeddings
    for sentence in sentences:
        embedding, classification, sentiment = model.predict_sentence(sentence)
        print(f"Sentence: {sentence}")
        print(f"Embedding: {embedding}\n")
        print(f"Classification: {classification.item()}\n")
        print(f"Sentiment: {sentiment:.4f}\n")


if __name__ == "__main__":
    lovely_tensors.monkey_patch()
    main()
