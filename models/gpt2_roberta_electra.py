import torch
from torch.optim import AdamW
from transformers import GPT2LMHeadModel, GPT2Tokenizer, get_linear_schedule_with_warmup
from transformers import ElectraForSequenceClassification, ElectraTokenizer, get_linear_schedule_with_warmup
from transformers import RobertaForSequenceClassification, RobertaTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm


class GPT2Emotion:
    """Class for implementing the GPT-2 emotion model"""

    def __init__(self, model_name='heegyu/gpt2-emotion'):
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def fit(self, train_loader, epochs, lr, weight_decay=0.01, fine_tune_last_layers=False):
        """Fine-tune the GPT-2 model on the given dataset."""

        # Freeze layers if fine-tuning only the last layers
        if fine_tune_last_layers:
            for param in self.model.base_model.parameters():
                param.requires_grad = False

        # Set up optimizer and scheduler
        optimizer = AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, weight_decay=weight_decay)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0

            for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
                b_input_ids, b_attention_mask, b_labels = tuple(t.to(self.device) for t in batch)
                self.model.zero_grad()

                # Forward pass
                outputs = self.model(input_ids=b_input_ids, attention_mask=b_attention_mask, labels=b_labels)
                loss = outputs.loss
                total_loss += loss.item()

                # Backward pass
                loss.backward()
                optimizer.step()
                scheduler.step()

            avg_train_loss = total_loss / len(train_loader)
            print(f"  Average training loss: {avg_train_loss}")

        return self

    def predict(self, data_loader, max_length=50, temperature=0.7, top_k=50, top_p=0.95):
        """Generate text predictions using the GPT-2 model."""

        self.model.eval()
        predictions = []

        with torch.no_grad():
            for batch in data_loader:
                b_input_ids, b_attention_mask, _ = tuple(t.to(self.device) for t in batch)

                # Generate text
                outputs = self.model.generate(
                    input_ids=b_input_ids,
                    attention_mask=b_attention_mask,
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=True,
                    no_repeat_ngram_size=2
                )
                generated_texts = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
                predictions.extend(generated_texts)

        return predictions

    def save_model(self, filepath):
        """Save the model's state dictionary to a file."""
        torch.save(self.model.state_dict(), filepath)



class Roberta:
    """Class for implementing the RoBERTa emotion model"""

    def __init__(self, model_name='Dimi-G/roberta-base-emotion'):
        self.model = RobertaForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def fit(self, train_loader, epochs, lr, weight_decay=0.01, fine_tune_last_layers=False):
        """Fine-tune the RoBERTa model on the given dataset."""

        # Freeze layers if fine-tuning only the last layers
        if fine_tune_last_layers:
            for param in self.model.roberta.parameters():
                param.requires_grad = False

        # Set up optimizer and scheduler
        optimizer = AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, weight_decay=weight_decay)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0

            for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
                b_input_ids, b_attention_mask, b_labels = tuple(t.to(self.device) for t in batch)
                self.model.zero_grad()

                # Forward pass
                outputs = self.model(input_ids=b_input_ids, attention_mask=b_attention_mask, labels=b_labels)
                loss = outputs.loss
                total_loss += loss.item()

                # Backward pass
                loss.backward()
                optimizer.step()
                scheduler.step()

            avg_train_loss = total_loss / len(train_loader)
            print(f"  Average training loss: {avg_train_loss}")

        return self

    def predict(self, data_loader):
        """Generate predictions using the RoBERTa model."""

        self.model.eval()
        predictions = []

        with torch.no_grad():
            for batch in data_loader:
                b_input_ids, b_attention_mask, _ = tuple(t.to(self.device) for t in batch)
                outputs = self.model(input_ids=b_input_ids, attention_mask=b_attention_mask)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                predictions.extend(preds.cpu().numpy())

        return predictions

    def save_model(self, filepath):
        """Save the model's state dictionary to a file."""
        torch.save(self.model.state_dict(), filepath)


class Electra:
    """Class for implementing the ELECTRA emotion model"""

    def __init__(self, model_name='mudogruer/electra-emotion'):
        self.model = ElectraForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = ElectraTokenizer.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def fit(self, train_loader, epochs, lr, weight_decay=0.01, fine_tune_last_layers=False):
        """Fine-tune the ELECTRA model on the given dataset."""

        # Freeze layers if fine-tuning only the last layers
        if fine_tune_last_layers:
            for param in self.model.electra.parameters():
                param.requires_grad = False

        # Set up optimizer and scheduler
        optimizer = AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, weight_decay=weight_decay)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0

            for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
                b_input_ids, b_attention_mask, b_labels = tuple(t.to(self.device) for t in batch)
                self.model.zero_grad()

                # Forward pass
                outputs = self.model(input_ids=b_input_ids, attention_mask=b_attention_mask, labels=b_labels)
                loss = outputs.loss
                total_loss += loss.item()

                # Backward pass
                loss.backward()
                optimizer.step()
                scheduler.step()

            avg_train_loss = total_loss / len(train_loader)
            print(f"  Average training loss: {avg_train_loss}")

        return self

    def predict(self, data_loader):
        """Generate predictions using the ELECTRA model."""

        self.model.eval()
        predictions = []

        with torch.no_grad():
            for batch in data_loader:
                b_input_ids, b_attention_mask, _ = tuple(t.to(self.device) for t in batch)
                outputs = self.model(input_ids=b_input_ids, attention_mask=b_attention_mask)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                predictions.extend(preds.cpu().numpy())

        return predictions

    def save_model(self, filepath):
        """Save the model's state dictionary to a file."""
        torch.save(self.model.state_dict(), filepath)