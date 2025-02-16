import os  # when loading file paths
import pandas as pd  # for lookup in annotation file
import spacy  # for tokenizer
import torch
from torch.nn.utils.rnn import pad_sequence  # pad batch
from torch.utils.data import DataLoader, Dataset
from PIL import Image  # Load img
import torchvision.transforms as transforms


# We want to convert text -> numerical values
# 1. We need a Vocabulary mapping each word to a index
# 2. We need to setup a Pytorch dataset to load the data
# 3. Setup padding of every batch (all examples should be
#    of same seq_len and setup dataloader)
# Note that loading the image is very easy compared to the text!

# Download with: python -m spacy download en
spacy_eng = spacy.load("en_core_web_sm")

seq_len = 100

class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(str(text))]

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word] = 1

                else:
                    frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]

def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0


class FlickrDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform=None, freq_threshold=5):
        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file)
        self.transform = transform

        # Get img, caption columns
        self.imgs = self.df["image"]
        self.captions = self.df["caption"]

        # Initialize vocabulary and build vocab
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions.tolist())


    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        caption = self.captions[index]
        img_id = self.imgs[index]

        self.pad_token = self.vocab.stoi["<PAD>"]



        img = Image.open(os.path.join(self.root_dir, img_id)).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        decoder_input = [self.vocab.stoi["<SOS>"]]
        decoder_input += self.vocab.numericalize(caption)
        decoder_input.append(self.vocab.stoi["<EOS>"])

        num_pad_tokens_input = seq_len - len(decoder_input)

        for _ in range(num_pad_tokens_input):
            decoder_input.append(self.pad_token)

        decoder_input = torch.tensor(decoder_input)

        tgt_mask = ((decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0))).clone().detach()

        label = []

        label = label + self.vocab.numericalize(caption)
        label.append(self.vocab.stoi["<EOS>"])

        num_pad_tokens_label = seq_len - len(label)

        for _ in range(num_pad_tokens_label):
            label.append(self.pad_token)

        label = torch.tensor(label)

        return img, decoder_input, tgt_mask, label


# class MyCollate:
#     def __init__(self):
#         # self.pad_idx = 1 # even i don't know why is did this
#         pass

#     def __call__(self, batch):
#         imgs = [item[0].unsqueeze(0) for item in batch]
#         imgs = torch.cat(imgs, dim=0)
#         # targets = [item[1] for item in batch]
#         # # targets = torch.cat(targets, dim=0)
#         # mask = [item[2] for item in batch]
#         # targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)

#         return imgs


def get_loader(
    root_folder,
    annotation_file,
    transform,
    batch_size=16,
    num_workers=8,
    shuffle=True,
    pin_memory=True,
):
    dataset = FlickrDataset(root_folder, annotation_file, transform=transform)

    pad_idx = dataset.vocab.stoi["<PAD>"]

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        
        pin_memory=pin_memory,
        
        drop_last=True
    )

    return loader, dataset


# if __name__ == "__main__":
#     transform = transforms.Compose(
#         [transforms.Resize((224, 224)), transforms.ToTensor(),]
#     )

#     loader, dataset = get_loader(
#         "flicker8k/Images", "flicker8k/captions.txt", transform=transform
#     )

#     for idx, (imgs, captions, tgt_mask, label) in enumerate(loader):
#         print(imgs.shape)
#         # print("Decoder input type:", type(captions))
#         print("Decoder Input Size:",captions.shape)
#         # print("Decoder mask Type:", type(tgt_mask))
#         print("Decoder mask shape:",tgt_mask.shape)

#         print("Label Shape:", label.shape)
#         break

