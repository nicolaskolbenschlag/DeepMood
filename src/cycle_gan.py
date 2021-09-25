import pandas as pd
import transformers
import torch
import sklearn.preprocessing
import numpy as np

MODEL_NAME = "bert-base-uncased"

class Dataset(torch.utils.data.Dataset):

    def __init__(self) -> None:
        self.tokenizer = transformers.BertTokenizer.from_pretrained(MODEL_NAME)
        df = pd.read_csv("data/IMDB Dataset.csv")
        self.texts = df["review"].to_list()
        self.labels = sklearn.preprocessing.LabelEncoder().fit_transform(df["sentiment"].to_list())        
    
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            max_length=32,
            add_special_tokens=True,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "targets": torch.tensor(label, dtype=torch.float)
        }

def save_model(model: torch.nn.Module, filename: str) -> None:
    torch.save(model.state_dict(), filename)

def load_model(model: torch.nn.Module, filename: str) -> None:
    model.load_state_dict(torch.load(filename))

# 1. generator: input -> seq2seq -> pos. sentiment output
# discriminator: sample pos. sentiment?
# 2. generator: bert encodings -> content similarity with encodings of input

class Discriminator(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        self.encoder = transformers.BertModel.from_pretrained(MODEL_NAME)
        self.dropout = torch.nn.Dropout(.3)
        self.out = torch.nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = output["pooler_output"]
        output = self.dropout(pooled_output)
        return self.out(output)

class Generator(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.autoencoder = transformers.EncoderDecoderModel.from_encoder_decoder_pretrained(MODEL_NAME, MODEL_NAME)
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

def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    epochs = 100
    
    dataset = Dataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64)

    generator = Generator().to(device)

    discriminator = Discriminator().to(device)
    optimizer_discriminator = transformers.AdamW(discriminator.parameters(), lr=2e-5, correct_bias=False)
    scheduler_discriminator = transformers.get_linear_schedule_with_warmup(optimizer_discriminator, num_warmup_steps=0, num_training_steps=len(dataloader) * epochs)
    loss_discriminator = torch.nn.MSELoss().to(device)

    gan = GAN(generator, discriminator)#.to(device)

    # NOTE training

    losses_discriminator = []
    losses_gan = []
    
    for epoch in range(1, epochs + 1):

        losses_discriminator += [[]]
        losses_gan += [[]]
        
        for data in dataloader:

            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            targets = data["targets"].to(device)

            # NOTE train discriminator (sentiment pos. or neg.?)
            outputs = discriminator(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_discriminator(outputs, targets)
            
            losses_discriminator[-1] += [loss]
            print(f"Epoch {epoch}: {loss.item()}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.)
            optimizer_discriminator.step()
            scheduler_discriminator.step()
            optimizer_discriminator.zero_grad()

            # TODO train GAN (generator followed by discriminator with generator weigths flozen)

    
    # NOTE evaluation
    losses_discriminator = np.array(losses_discriminator).mean(axis=1)
    
    # save_model(gan, "gan.pth")

if __name__ == "__main__":
    main()