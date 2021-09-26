import pandas as pd
import transformers
import torch
import numpy as np

MODEL_NAME = "bert-base-uncased"

class Dataset(torch.utils.data.Dataset):

    def __init__(self) -> None:
        self.tokenizer = transformers.BertTokenizer.from_pretrained(MODEL_NAME)
        df = pd.read_csv("data/IMDB Dataset.csv")
        self.texts = df["review"].to_list()
        # self.labels = sklearn.preprocessing.LabelEncoder().fit_transform(df["sentiment"].to_list())        
        self.labels = [1. if y == "positive" else 0. for y in df["sentiment"].to_list()]
    
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text=text,
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
        encoding = output["pooler_output"]
        output = self.dropout(encoding)
        return torch.sigmoid(self.out(output)), encoding
    
    def unfreeze(self):
        self.train()
        self.encoder.eval()
    
    def freeze(self):
        self.eval()
    
class Generator(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder_decoder = transformers.EncoderDecoderModel.from_encoder_decoder_pretrained(MODEL_NAME, MODEL_NAME)    
    
    def forward(self, input_ids):
        output = self.encoder_decoder(input_ids=input_ids, decoder_input_ids=input_ids)
        encoding_input = output["encoder_last_hidden_state"]
        
        # TODO fix this?! (maybe encode both original and generated afterwards)
        encoding_input = encoding_input[:,-1,:]
        
        generated = output["logits"]
        return generated, encoding_input

    def unfreeze(self):
        self.train()
        # NOTE encoder weights frozen (reduce required computing resources)
        self.encoder_decoder.get_encoder().eval()

class GAN(torch.nn.Module):

    def __init__(self, generator: torch.nn.Module, discriminator: torch.nn.Module):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
    
    def un_freeze_weights(self):
        self.generator.unfreeze()
        self.discriminator.freeze()
    
    def forward(self, input_ids, attention_mask):
        generated, encoding_input = self.generator(input_ids=input_ids)        
        generated = generated.argmax(dim=2)
        sentiment, encoding_output = self.discriminator(generated, attention_mask=attention_mask)
        return sentiment, encoding_input, encoding_output, generated

def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    epochs = 3
    batch_size = 64
    
    dataset = Dataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    generator = Generator().to(device)

    discriminator = Discriminator().to(device)
    optimizer_discriminator = transformers.AdamW(discriminator.parameters(), lr=2e-5, correct_bias=False)
    scheduler_discriminator = transformers.get_linear_schedule_with_warmup(optimizer_discriminator, num_warmup_steps=0, num_training_steps=len(dataloader) * epochs)
    loss_fn_discriminator = torch.nn.MSELoss().to(device)

    gan = GAN(generator, discriminator).to(device)
    optimizer_gan = transformers.AdamW(gan.parameters(), lr=2e-5, correct_bias=False)
    scheduler_gan = transformers.get_linear_schedule_with_warmup(optimizer_gan, num_warmup_steps=0, num_training_steps=len(dataloader) * epochs)
    loss_fn_gan_content = torch.nn.CosineSimilarity().to(device)

    # NOTE training
    losses_discriminator = []
    losses_gan = []
    
    for epoch in range(1, epochs + 1):

        losses_discriminator += [[]]
        losses_gan += [[]]
        
        for i_batch, data in enumerate(dataloader, 1):

            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            targets = data["targets"].unsqueeze(dim=1).to(device)

            # NOTE train discriminator (sentiment pos. or neg.?)
            discriminator.unfreeze()
            outputs = discriminator(
                input_ids=input_ids,
                attention_mask=attention_mask
            )[0]
            loss_discriminator = loss_fn_discriminator(outputs, targets)
            losses_discriminator[-1] += [loss_discriminator.item()]

            loss_discriminator.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.)
            optimizer_discriminator.step()
            scheduler_discriminator.step()
            optimizer_discriminator.zero_grad()

            # NOTE train GAN (generator followed by discriminator with generator weigths flozen)
            negative_sentiment_mask = (targets == 0.).squeeze(dim=1)
            gan.un_freeze_weights()
            outputs = gan(
                input_ids=input_ids[negative_sentiment_mask],
                attention_mask=attention_mask[negative_sentiment_mask]
            )

            # NOTE print generated sentence
            n = np.random.randint(0, negative_sentiment_mask.sum())
            print(f"Original sentence: {dataset.tokenizer.decode(input_ids[negative_sentiment_mask][n])}")
            print(f"Predicted sentiment: {outputs[0][n].item()}")
            print(f"Generated sentence: {dataset.tokenizer.decode(outputs[3][n])}")

            targets_gan = torch.tensor([1.] * negative_sentiment_mask.sum(), dtype=torch.float).unsqueeze(dim=1).to(device)
            loss_gan_sentiment = loss_fn_discriminator(outputs[0], targets_gan)
            encoding_input = outputs[1]
            encoding_output = outputs[2]
            
            content_similarity = loss_fn_gan_content(encoding_input, encoding_output)
            loss_gan_content = - content_similarity.mean()
            alpha = .5
            loss_gan = alpha * loss_gan_sentiment + (1. - alpha) * loss_gan_content
            
            losses_gan[-1] += [loss_gan.item()]

            loss_gan.backward()
            torch.nn.utils.clip_grad_norm_(gan.parameters(), max_norm=1.)
            optimizer_gan.step()
            scheduler_gan.step()
            optimizer_gan.zero_grad()

            print(f"Epoch {epoch} [{i_batch}] - discriminator [MSE] {loss_discriminator.item()}, GAN [MSE] {loss_gan_sentiment.item()}, GAN [Cosine] {loss_gan_content}")

            if i_batch == 10:
                break

    
    # NOTE evaluation
    losses_discriminator = np.array(losses_discriminator).mean(axis=1)
    
    # save_model(gan, "gan.pth")

    # TODO check weights freezing

if __name__ == "__main__":
    main()