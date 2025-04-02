import torch
from torch.optim import AdamW
from transformers import BertForSequenceClassification,AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from transformers import AutoTokenizer
from tqdm import tqdm


class Bert:
    """Class for implementing the three BERT models"""

    def __init__(self, model_version='bhadresh-savani/bert-base-uncased-emotion'):
        # First load the model with original number of classes
        self.model = BertForSequenceClassification.from_pretrained(model_version, output_attentions=True)
        
        # Update the model's configuration for 7 classes
        self.model.config.num_labels = 7
        
        # Update the model's internal state for 7 classes
        self.model.num_labels = 7
        self.model.classifier = torch.nn.Linear(self.model.config.hidden_size, 7)
        self.model.loss_fct = torch.nn.CrossEntropyLoss()
        
        # Move model to device
        self.model.to('cuda' if torch.cuda.is_available() else 'cpu')

    def fit(self, train_loader, epochs, lr, weight_decay=0.01, fine_tune_last_layers=False):
        """Train the model with improved parameters"""
        
        # Only freeze BERT layers if fine-tuning last layers
        if fine_tune_last_layers:
            for param in self.model.bert.parameters():
                param.requires_grad = False

        # Set up optimizer with improved parameters
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-08
        )
        
        # Calculate total steps for scheduler
        total_steps = len(train_loader) * epochs
        
        # Set up scheduler with warmup
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=total_steps // 10,  # 10% warmup
            num_training_steps=total_steps
        )

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0

            for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
                b_input_ids, b_input_mask, b_labels = tuple(t.to(self.model.device) for t in batch)
                self.model.zero_grad()

                outputs = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
                loss = outputs.loss
                total_loss += loss.item()
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()

            avg_train_loss = total_loss / len(train_loader)
            print(f"  Average training loss: {avg_train_loss}")

        return self

    def predict(self, data_loader, return_attentions=False):

        self.model.eval()
        y_pred = []
        all_attentions = [] if return_attentions else None

        with torch.no_grad():
            for batch in data_loader:
                b_input_ids, b_input_mask, b_labels = tuple(t.to(self.model.device) for t in batch)
                outputs = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
                logits = outputs.logits
                y_pred.extend(logits.argmax(dim=1).cpu().numpy())

                if return_attentions:
                    attentions = outputs.attentions
                    all_attentions.extend(attentions)

        return (y_pred, all_attentions) if return_attentions else y_pred

    def save_model(self, filepath):
        torch.save(self.model.state_dict(), filepath)

class DistilBert:
    """Class for implementing DistilBERT with identical structure to original BERT class"""

    def __init__(self, model_version='bhadresh-savani/distilbert-base-uncased-emotion'):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_version, output_attentions=True, num_labels=7)
        self.model.classifier = torch.nn.Linear(self.model.config.hidden_size, 7)
        self.model.to('cuda' if torch.cuda.is_available() else 'cpu')

    def fit(self, train_loader, epochs, lr, weight_decay=0.01, fine_tune_last_layers=False):
        """Identical training method structure"""
        
        if fine_tune_last_layers:
            for param in self.model.distilbert.parameters():  # Changed from model.bert
                param.requires_grad = False

        # Identical optimizer setup
        optimizer = AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, weight_decay=weight_decay)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0

            for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
                b_input_ids, b_input_mask, b_labels = tuple(t.to(self.model.device) for t in batch)
                self.model.zero_grad()

                # Only change: removed token_type_ids as DistilBERT doesn't use them
                outputs = self.model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
                loss = outputs.loss
                total_loss += loss.item()
                loss.backward()

                optimizer.step()
                scheduler.step()

            avg_train_loss = total_loss / len(train_loader)
            print(f"  Average training loss: {avg_train_loss}")

        return self

    def predict(self, data_loader, return_attentions=False):
        """Identical prediction method structure"""
        
        self.model.eval()
        y_pred = []
        all_attentions = [] if return_attentions else None

        with torch.no_grad():
            for batch in data_loader:
                b_input_ids, b_input_mask, b_labels = tuple(t.to(self.model.device) for t in batch)
                
                # Only change: removed token_type_ids
                outputs = self.model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
                logits = outputs.logits
                y_pred.extend(logits.argmax(dim=1).cpu().numpy())

                if return_attentions:
                    attentions = outputs.attentions
                    all_attentions.extend(attentions)

        return (y_pred, all_attentions) if return_attentions else y_pred

    def save_model(self, filepath):
        """Identical save method"""
        torch.save(self.model.state_dict(), filepath)