import torch
import torch.nn as nn
import torch.nn.functional as F
from Vit_Model import VitModel
from transformer_model import build_transformer
from PIL import Image
import torchvision.transforms as transforms
from data_loader import causal_mask



batch_size = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'

device = torch.device(device)
# print(device)

def print_examples(model, device, dataset):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    model.eval()
    test_img1 = transform(Image.open("flicker8k/Images/47871819_db55ac4699.jpg").convert("RGB")).unsqueeze(
        0
    )
    print("Example 1 CORRECT: Dog on a beach by the ocean")
    print(
        "Example 1 OUTPUT: "
        + " ".join(model.Caption_Generation(test_img1.to(device), dataset.vocab))
    )
    test_img2 = transform(
        Image.open("flicker8k/Images/3711030008_3872d0b03f.jpg").convert("RGB")
    ).unsqueeze(0)
    print("Example 2 CORRECT: Child holding red frisbee outdoors")
    print(
        "Example 2 OUTPUT: "
        + " ".join(model.Caption_Generation(test_img2.to(device), dataset.vocab))
    )
    test_img3 = transform(Image.open("flicker8k/Images/3729405438_6e79077ab2.jpg").convert("RGB")).unsqueeze(
        0
    )
    print("Example 3 CORRECT: Bus driving by parked cars")
    print(
        "Example 3 OUTPUT: "
        + " ".join(model.Caption_Generation(test_img3.to(device), dataset.vocab))
    )
    test_img4 = transform(
        Image.open("flicker8k/Images/3730011219_588cdc7972.jpg").convert("RGB")
    ).unsqueeze(0)
    print("Example 4 CORRECT: A small boat in the ocean")
    print(
        "Example 4 OUTPUT: "
        + " ".join(model.Caption_Generation(test_img4.to(device), dataset.vocab))
    )
    test_img5 = transform(
        Image.open("flicker8k/Images/3724718895_bd03f4a4dc.jpg").convert("RGB")
    ).unsqueeze(0)
    print("Example 5 CORRECT: A cowboy riding a horse in the desert")
    print(
        "Example 5 OUTPUT: "
        + " ".join(model.Caption_Generation(test_img5.to(device), dataset.vocab))
    )




class VisionWithTransformer(nn.Module):
    def __init__(self,vocab_size):
        super(VisionWithTransformer,self).__init__()


        self.vocab_size = vocab_size

        self.vision_model = VitModel()

        self.transformer_model = build_transformer(tgt_vocab_size=self.vocab_size,tgt_seq_len=100,d_model=512,N=12,h=8,dropout=0.1,d_ff=2048)

        self.params = list(self.vision_model.parameters()) + list(self.transformer_model.parameters())

    def forward(self,imgs,encoder_mask,decoder_input,decoder_mask):
        
        


        encoder_input = self.vision_model(imgs)

        encoder_output = self.transformer_model.encode(encoder_input,encoder_mask)
        decoder_output = self.transformer_model.decode(encoder_output,encoder_mask,decoder_input,decoder_mask)
        proj_output = self.transformer_model.project(decoder_output)


        return proj_output
    

    def Caption_Generation(self,imgs,vocab,max_length = 50):

        sos_idx = vocab.stoi["<SOS>"]
        eos_idx = vocab.stoi["<EOS>"]

        # print("Image Size:",imgs.shape)

        input = []
        input = torch.tensor(input).to(device)
        for _ in range(batch_size):
            input=torch.cat((input,imgs),dim=0)

        # print("Input Size:",input.shape)
        source = self.vision_model(input)
        # print("Source Size:",source.shape)

        source = source[0,:,:].unsqueeze(0)
        # print("Source Size:",source.shape)
        # print("Source Type:", type(source))
        encoder_mask = None
        decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
        # print("Decoder Input Shape:",decoder_input.shape)
        encoder_output = self.transformer_model.encode(source,encoder_mask)
        # print("Encoder Output Size:",encoder_output.shape)
        while True:
            if decoder_input.size(1) == max_length:
                break

            # build mask for target
            decoder_mask = causal_mask(decoder_input.size(1)).type_as(source).to(device)
            # print("Decoder Maskk size:", decoder_mask.shape)
            decoder_output = self.transformer_model.decode(encoder_output,encoder_mask,decoder_input,decoder_mask)
            # print("Decoder output size:", decoder_output.shape)
            

            prob = self.transformer_model.project(decoder_output[:,-1])
            # print("Final output size:", prob.size)
            _, next_word = torch.max(prob, dim=1)

            decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
            )



            if next_word == eos_idx:
                break
        # print(decoder_input.int())
        return [vocab.itos[int(idx)] for idx in decoder_input.view(-1).tolist()]


            

