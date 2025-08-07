
import pandas as pd
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from sklearn.preprocessing import LabelEncoder
from config import DATA_DIR
import torch

class CovidTweetDataset(Dataset): # Dataset Class
    def __init__(self, dataframe, tokenizer,max_length):
        # Extract the 'text_combined' and 'label' columns from the DataFrame
        self.texts = dataframe['tweets'].tolist()
        self.labels = dataframe['Sentiment'].tolist()
        self.tokenizer = tokenizer # Tokenizer for text processing
        self.max_length = max_length  # Maximum length for padding/truncation
    def __len__(self): #Returns the total number of samples in the dataset.
        # This method is required for PyTorch's DataLoader to work !!
        return len(self.texts)

    def __getitem__(self, idx): #Retrieves a single data sample and its label at the specified index.
        text = self.texts[idx]
        label = self.labels[idx]

        if not isinstance(text, str):
            text = str(text)  # Ensure text is a string

        encoding = self.tokenizer(
            [text],  # Make it a batch of one
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
            return_token_type_ids=False
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),  # Shape: [512]
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_dataframes():
    train_df = pd.read_csv(DATA_DIR/"processed"/ "train_data.csv")
    test_df = pd.read_csv(DATA_DIR/"processed"/ "test_data.csv")
    return train_df, test_df


def prepare_dataset(tokenizer,max_length):
    train_df, test_df = load_dataframes()
    train_ds = CovidTweetDataset(train_df, tokenizer, max_length)
    test_ds = CovidTweetDataset(test_df, tokenizer, max_length)
    labels = train_df['Sentiment'].tolist()
    return train_ds, test_ds, labels


