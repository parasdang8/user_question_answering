# user_question_answering.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForQuestionAnswering
import torch
from torch.utils.data import DataLoader, Dataset

class QADataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        context = str(self.data.context[index])
        question = str(self.data.question[index])
        answer = str(self.data.answer[index])

        encoding = self.tokenizer.encode_plus(
            question,
            context,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        inputs = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'start_positions': torch.tensor(self.data.start_position[index], dtype=torch.long),
            'end_positions': torch.tensor(self.data.end_position[index], dtype=torch.long)
        }
        return inputs

def train_model(model, data_loader, optimizer, device, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0

    for data in data_loader:
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        token_type_ids = data['token_type_ids'].to(device)
        start_positions = data['start_positions'].to(device)
        end_positions = data['end_positions'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            start_positions=start_positions,
            end_positions=end_positions
        )

        loss = outputs[0]
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    return np.mean(losses)

def eval_model(model, data_loader, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for data in data_loader:
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            token_type_ids = data['token_type_ids'].to(device)
            start_positions = data['start_positions'].to(device)
            end_positions = data['end_positions'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                start_positions=start_positions,
                end_positions=end_positions
            )

            loss = outputs[0]
            losses.append(loss.item())

    return np.mean(losses)

def main():
    # Load and preprocess data
    df = pd.read_csv('qa_dataset.csv')

    # Tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')

    # Data preparation
    dataset = QADataset(df, tokenizer, max_len=512)
    train_data, val_data = train_test_split(dataset, test_size=0.1)
    train_loader = DataLoader(train_data, batch_size=8)
    val_loader = DataLoader(val_data, batch_size=8)

    # Training setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_loader) * 3
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    for epoch in range(3):
        train_loss = train_model(model, train_loader, optimizer, device, scheduler, len(train_data))
        print(f"Epoch {epoch+1}/{3} | Train Loss: {train_loss}")

        val_loss = eval_model(model, val_loader, device, len(val_data))
        print(f"Epoch {epoch+1}/{3} | Validation Loss: {val_loss}")

if __name__ == "__main__":
    main()
