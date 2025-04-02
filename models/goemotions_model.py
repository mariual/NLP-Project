from transformers import BertForSequenceClassification
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import torch.nn as nn

class GoEmotionsBert:
    def __init__(self, model_name="justin871030/bert-base-uncased-goemotions-original-finetuned", num_labels=28):
        from transformers import BertForSequenceClassification
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, problem_type="multi_label_classification")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def fit(self, train_loader, epochs=3, lr=2e-5, weight_decay=0.01):
        criterion = nn.BCEWithLogitsLoss()
        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=epochs*len(train_loader))

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
                input_ids, attention_mask, labels = (t.to(self.device) for t in batch)
                self.model.zero_grad()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")

    def predict(self, data_loader, threshold=0.5):
        self.model.eval()
        all_preds = []
        with torch.no_grad():
            for batch in data_loader:
                input_ids, attention_mask, _ = (t.to(self.device) for t in batch)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.sigmoid(outputs.logits)
                preds = (probs > threshold).int().cpu()
                all_preds.extend(preds)
        return torch.stack(all_preds)
