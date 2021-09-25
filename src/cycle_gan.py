import pandas as pd
import transformers
import torch

def read_dataset(filename: str) -> object:
    df = pd.read_csv(filename)
    X, y = df["review"].to_list(), df["sentiment"].to_list()
    return X, y

def save_model(model: torch.nn.Module, filename: str) -> None:
    torch.save(model.state_dict(), filename)

def load_model(model: torch.nn.Module, filename: str) -> None:
    model.load_state_dict(torch.load(filename))

# 1. generator: input -> seq2seq -> pos. sentiment output
# discriminator: sample pos. sentiment?
# 2. generator: bert encodings -> content similarity with encodings of input

# https://curiousily.com/posts/sentiment-analysis-with-bert-and-hugging-face-using-pytorch-and-python/
class Discriminator(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        modelname = "bert-base-uncased"
        self.encoder = transformers.BertModel.from_pretrained(modelname)
        self.dropout = torch.nn.Dropout(.3)
        self.out = torch.nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        output = self.drop(pooled_output)
        return self.out(output)

class Generator(torch.nn.Module):

    def __init__(self):
        super().__init__()
        modelname = "bert-base-uncased"
        self.autoencoder = transformers.EncoderDecoderModel.from_encoder_decoder_pretrained(modelname, modelname)
        # TODO encoder weights frozen (reduce required computing resources)?
    
    def forward(self, input_ids):
        return self.autoencoder(input_ids)

class GAN(torch.nn.Module):

    def __init__(self, generator: torch.nn.Module, discriminator: torch.nn.Module):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
    
    def forward(self, input_ids):
        return self.discriminator(self.generator(input_ids))

# class SimilarityMetric(torch.nn.Module):
#     def __init__(self):
#         super().__init__()

def main() -> None:
    X, y = read_dataset("data/IMDB Dataset.csv")
    print(f"X: {len(X)}, y: {len(y)}")

    generator = Generator()
    discriminator = Discriminator()
    gan = GAN(generator, discriminator)

    epochs = 100
    batch_size = 64
    for epoch in range(epochs):
        
        for i_batch in range(0, len(X), batch_size):

            # TODO train discriminator (sentiment pos. or neg.?)

            # TODO train GAN (generator followed by discriminator with generator weigths flozen)

            pass
    
    save_model(gan, "gan.pth")

if __name__ == "__main__":
    main()