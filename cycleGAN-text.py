import pandas as pd
import transformers
import torch

def read_dataset(filename: str) -> object:
    df = pd.read_csv(filename)
    X, y = df["review"].to_list(), df["sentiment"].to_list()
    return X, y

# 1. generator: input -> seq2seq -> pos. sentiment output
# discriminator: sample pos. sentiment?
# 2. generator: bert encodings -> content similarity with encodings of input

# https://curiousily.com/posts/sentiment-analysis-with-bert-and-hugging-face-using-pytorch-and-python/
class Discriminator(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        modelname = "bert-base-uncased"
        self.bert = transformers.BertModel.from_pretrained(modelname)
        self.dropout = torch.nn.Dropout(.3)
        self.out = torch.nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        output = self.drop(pooled_output)
        return self.out(output)

class Generator(torch.nn.Module):

    def __init__(self):
        super().__init__()
        modelname = "bert-base-uncased"
        self.bert = transformers.EncoderDecoderModel.from_encoder_decoder_pretrained(modelname, modelname)
    
    def forward(self, input_ids):
        return self.bert(input_ids)

class SimilarityMetric(torch.nn.Module):
    
    def __init__(self):
        super().__init__()

def main() -> None:
    X, y = read_dataset("data/IMDB Dataset.csv")
    print(f"X: {len(X)}, y: {len(y)}")

    discriminator = Discriminator()
    generator = Generator()

if __name__ == "__main__":
    main()