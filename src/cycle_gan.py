import pandas as pd
import transformers
import torch
import numpy as np

MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 32#128

class Dataset(torch.utils.data.Dataset):

    def __init__(self) -> None:
        self.tokenizer = transformers.BertTokenizer.from_pretrained(MODEL_NAME)
        df = pd.read_csv("data/IMDB Dataset.csv")
        self.texts = df["review"].to_list()
        self.labels = [1. if y == "positive" else 0. for y in df["sentiment"].to_list()]
    
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text=text,
            max_length=MAX_LENGTH,
            padding="max_length",
            add_special_tokens=True,
            return_token_type_ids=False,
            truncation=True,
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

class Generator(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder_decoder = transformers.EncoderDecoderModel.from_encoder_decoder_pretrained(MODEL_NAME, MODEL_NAME, max_length=MAX_LENGTH)
    
    def forward(self, input_ids):
        generated = self.encoder_decoder.generate(input_ids=input_ids, decoder_start_token_id=self.encoder_decoder.config.decoder.pad_token_id)
        return generated

    def unfreeze(self):
        # self.train()
        for param in self.parameters():
            param.requires_grad = True
        # NOTE encoder weights frozen (reduce required computing resources)
        # self.encoder_decoder.get_encoder().eval()
        for param in self.encoder_decoder.get_encoder().parameters():
            param.requires_grad = False
    
    def freeze(self):
        # self.eval()
        for param in self.parameters():
            param.requires_grad = False

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
        return torch.sigmoid(self.out(output))
    
    def unfreeze(self):
        # self.train()
        for param in self.parameters():
            param.requires_grad = True
        # self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
        
    def freeze(self):
        # self.eval()
        for param in self.parameters():
            param.requires_grad = False

class GAN(torch.nn.Module):

    def __init__(self, generator: torch.nn.Module, discriminator: torch.nn.Module):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
    
    def un_freeze_weights(self):
        self.generator.unfreeze()
        self.discriminator.freeze()
    
    def forward(self, input_ids, attention_mask):
        generated = self.generator(input_ids=input_ids)
        # TODO is this attention_mask correct?
        sentiment = self.discriminator(generated, attention_mask=attention_mask)
        return generated, sentiment

class ContentSimilarityLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = transformers.BertModel.from_pretrained(MODEL_NAME)
        self.encoder.eval()
    
    def forward(self, input_ids_1, input_ids_2):
        encoding_1 = self.encoder(input_ids=input_ids_1)["pooler_output"]
        encoding_2 = self.encoder(input_ids=input_ids_2)["pooler_output"]
        similarity = torch.nn.functional.cosine_similarity(encoding_1, encoding_2)
        similarity = (- similarity + 1) / 2
        return similarity.mean()

def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    epochs = 10
    batch_size = 64
    
    dataset = Dataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    generator = Generator().to(device)

    discriminator = Discriminator().to(device)
    optimizer_discriminator = transformers.AdamW(discriminator.parameters(), lr=2e-5)#, correct_bias=False)
    scheduler_discriminator = transformers.get_linear_schedule_with_warmup(optimizer_discriminator, num_warmup_steps=0, num_training_steps=len(dataloader) * epochs)
    loss_fn_discriminator = torch.nn.MSELoss().to(device)

    gan = GAN(generator, discriminator).to(device)
    optimizer_gan = transformers.AdamW(gan.parameters(), lr=2e-5)#, correct_bias=False)
    scheduler_gan = transformers.get_linear_schedule_with_warmup(optimizer_gan, num_warmup_steps=0, num_training_steps=len(dataloader) * epochs)
    loss_fn_gan_content = ContentSimilarityLoss().to(device)

    # TODO rename variables (losses, ...) more specificly
    # NOTE training
    losses_discriminator, losses_gan = [], []    
    for epoch in range(1, epochs + 1):

        losses_discriminator += [[]]
        losses_gan += [[]]

        latest_generated_input_ids, latest_attention_mask = None, None
        
        for i_batch, data in enumerate(dataloader, 1):

            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            targets_sentiment = data["targets"].unsqueeze(dim=1).to(device)

            # NOTE train discriminator: sentiment pos. or neg.?
            discriminator.unfreeze()
            outputs = discriminator(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            loss_discriminator = loss_fn_discriminator(outputs, targets_sentiment)

            # TODO how to balance!? (now more training for output 0.)
            # NOTE train discriminator: sample fake or real?
            if not latest_generated_input_ids is None:
                targets_fake = torch.tensor([0.] * len(latest_generated_input_ids), dtype=torch.float).unsqueeze(dim=1).to(device)
                outputs = discriminator(input_ids=latest_generated_input_ids, attention_mask=latest_attention_mask)
                loss_discriminator_real_fake = loss_fn_discriminator(outputs, targets_fake)
                loss_discriminator = loss_discriminator + loss_discriminator_real_fake

            losses_discriminator[-1] += [loss_discriminator.item()]

            loss_discriminator.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.)
            optimizer_discriminator.step()
            scheduler_discriminator.step()
            optimizer_discriminator.zero_grad()

            # NOTE train GAN (generator followed by discriminator with generator weigths flozen)
            negative_sentiment_mask = (targets_sentiment == 0.).squeeze(dim=1)
            
            gan.un_freeze_weights()
            generated, sentiment = gan(
                input_ids=input_ids[negative_sentiment_mask],
                attention_mask=attention_mask[negative_sentiment_mask]
            )

            latest_generated_input_ids = generated
            latest_attention_mask = attention_mask[negative_sentiment_mask]

            # NOTE print generated sentence
            n = np.random.randint(0, negative_sentiment_mask.sum())
            print(f"Original sentence:\t{dataset.tokenizer.decode(input_ids[negative_sentiment_mask][n])} [sentiment: {round(sentiment[n].item(), 4)}]")
            print(f"Generated sentence:\t{dataset.tokenizer.decode(generated[n])}")

            targets_gan = torch.tensor([1.] * negative_sentiment_mask.sum(), dtype=torch.float).unsqueeze(dim=1).to(device)
            loss_gan_sentiment = loss_fn_discriminator(sentiment, targets_gan)
            loss_gan_content = loss_fn_gan_content(input_ids[negative_sentiment_mask], generated)
            
            alpha = .25
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
    losses_discriminator_epoch = np.array(losses_discriminator).mean(axis=1)
    
    # save_model(gan, "gan.pth")

if __name__ == "__main__":
    main()