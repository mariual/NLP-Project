import torch
from torch.optim import AdamW
from transformers import GPT2LMHeadModel, GPT2Tokenizer, get_linear_schedule_with_warmup
from transformers import ElectraForSequenceClassification, ElectraTokenizer, get_linear_schedule_with_warmup
from transformers import RobertaForSequenceClassification, RobertaTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import random
import logging
import pandas as pd
from transformers import GPT2ForSequenceClassification
import numpy as np
import torch.nn as nn


# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GPT2Emotion:
    """Class for implementing the GPT-2 emotion model."""

    def __init__(self, model_name='heegyu/gpt2-emotion', num_labels=6, seed=42):
        """Initialize the GPT-2 model and tokenizer for classification."""
        self.seed = seed
        self._set_seed()

        # Charger le modèle et le tokenizer GPT-2 pour la classification
        self.model = GPT2ForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)

        # Ajouter un token de padding si nécessaire (GPT-2 n'en a pas par défaut)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.model.resize_token_embeddings(len(self.tokenizer))

        # Déplacer le modèle sur le GPU si disponible
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def _set_seed(self):
        """Set seed for reproducibility."""
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def fit(self, train_loader, epochs, lr, weight_decay=0.01, fine_tune_last_layers=False):
        """Fine-tune the GPT-2 model on the given dataset for classification."""

        # Geler les couches si on ne fine-tune que les dernières couches
        if fine_tune_last_layers:
            for param in self.model.base_model.parameters():
                param.requires_grad = False
            for param in self.model.transformer.h[-1].parameters():  # Dernière couche de transformation
                param.requires_grad = True
            for param in self.model.score.parameters():  # Couche de classification
                param.requires_grad = True
        else:
            # Dégeler tous les paramètres si fine_tune_last_layers=False
            for param in self.model.parameters():
                param.requires_grad = True

        # Configurer l'optimiseur
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-08
        )
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        # Fonction de perte pour la classification
        criterion = nn.CrossEntropyLoss()

        # Boucle d'entraînement
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0

            for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
                b_input_ids, b_attention_mask, b_labels = tuple(t.to(self.device) for t in batch)

                # Forward pass
                outputs = self.model(input_ids=b_input_ids, attention_mask=b_attention_mask)
                logits = outputs.logits  # Logits de classification

                # Calcul de la perte
                loss = criterion(logits, b_labels)
                total_loss += loss.item()

                # Backward pass
                loss.backward()
                optimizer.step()
                scheduler.step()

            # Calcul de la perte moyenne
            avg_train_loss = total_loss / len(train_loader)
            logging.info(f"Epoch {epoch + 1}: Average training loss = {avg_train_loss}")

        return self

    def predict(self, data_loader):
        """Generate class predictions using the GPT-2 model."""

        self.model.eval()
        predictions = []

        with torch.no_grad():
            for batch in data_loader:
                b_input_ids, b_attention_mask, _ = tuple(t.to(self.device) for t in batch)

                # Forward pass
                outputs = self.model(input_ids=b_input_ids, attention_mask=b_attention_mask)
                logits = outputs.logits

                # Obtenir les classes prédites
                preds = torch.argmax(logits, dim=-1)
                predictions.extend(preds.cpu().numpy())

        return predictions
    def save_model(self, filepath):
        """Save the model's state dictionary to a file."""
        torch.save(self.model.state_dict(), filepath)

    def load_model(self, filepath):
        """Load the model's state dictionary from a file."""
        self.model.load_state_dict(torch.load(filepath))
        self.model.to(self.device)

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
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-08
        )
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


# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Electra:
    """Class for implementing the ELECTRA emotion model."""

    def __init__(self, model_name='mudogruer/electra-emotion', seed=42):
        """Initialize the ELECTRA model and tokenizer."""
        self.seed = seed
        self._set_seed()

        self.model = ElectraForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = ElectraTokenizer.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def _set_seed(self):
        """Set seed for reproducibility."""
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def fit(self, train_loader, epochs=3, lr=5e-5, weight_decay=0.01, fine_tune_last_layers=False):
        """Fine-tune the ELECTRA model on the given dataset."""
        if fine_tune_last_layers:
            for param in self.model.electra.parameters():
                param.requires_grad = False

        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-08
        )

        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0

            for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
                b_input_ids, b_attention_mask, b_labels = tuple(t.to(self.device) for t in batch)
                self.model.zero_grad()

                outputs = self.model(input_ids=b_input_ids, attention_mask=b_attention_mask, labels=b_labels)
                loss = outputs.loss
                total_loss += loss.item()

                loss.backward()
                optimizer.step()
                scheduler.step()

            avg_train_loss = total_loss / len(train_loader)
            logging.info(f"Epoch {epoch + 1}: Average training loss = {avg_train_loss}")

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
