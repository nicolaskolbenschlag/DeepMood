# DeepMood

Author: Nicolas Kolbenschlag (University of Augsburg)

This project aims to transfer approaches from **Neural Style Transfer** from images to text.

## 1st approach: Neural style transfer

The file src/neural_style_transfer.py tries to transfer the ideas from [Neural style transfer](https://arxiv.org/abs/1508.06576) from images to Natural Language Processing.

![architecture](https://github.com/nicolaskolbenchlag/DeepMood/blob/main/images/Architecture.png)

We compute the gradients of the encoded input sentence and optimize it.

### Run neural style transfer for text

```shell
$python neural_sytle_transfer.py
```

## 2nd approach: CycleGAN

Here, we adopt the architectore from [CycleGAN](https://junyanz.github.io/CycleGAN/) for text paraphrasing.
We train a generator for seq2seq modelling and a discriminator to evaluate the generator's output like **CycleGAN**.

1. generator: input -> seq2seq -> pos. sentiment output
2. discriminator: sample pos. sentiment?
3. generator: bert encodings -> content similarity with encodings of input

### Run CycleGAN for text

```shell
$pip install -r requirements.txt
```

```shell
$python cycle_gan.py
```

## Usage

* The service could be used be messaging app providers (like Signal, WhatsApp or Gmail) to provide their customers suggestions on how to rewrite their messages.
* Each company / service that interacts with customers by language wants to be as friendly and positive as possible.

## Credits

* Nice [tutorial](https://curiousily.com/posts/sentiment-analysis-with-bert-and-hugging-face-using-pytorch-and-python/) on BERT with HuggingFace
