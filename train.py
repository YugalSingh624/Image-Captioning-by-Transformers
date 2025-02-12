import torch 
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from model_imagecaptioning import VisionWithTransformer,print_examples
from data_loader import get_loader
from tqdm import tqdm

# Ensure that the script runs properly on Windows
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using Device:", device)
    if device == "cuda":
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        print(f"Device memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3} GB")
    else:
        print("NOTE: If you have a GPU, consider using it for training.")

    device = torch.device(device)

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    # DataLoader
    loader, dataset = get_loader(
        "flicker8k/Images", "flicker8k/captions.txt", transform=transform
    )

    vocab_size = len(dataset.vocab)
    print("Total number of words in vocab:", vocab_size)
    # print("vOCAB:", dataset.vocab)
    model = VisionWithTransformer(vocab_size=vocab_size).to(device)

    params = model.parameters()

    loss_fn = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"], label_smoothing=0.1).to(device)
    optimizer = optim.AdamW(params, lr=3e-4, weight_decay=1e-4, betas=(0.9, 0.98))
    print("Total number of learnable parameters:",sum(p.numel() for p in model.parameters() if p.requires_grad))

    # model_filename = "saved_model.pt"
    # if model_filename:
    #     print(f'Preloading model {model_filename}')
    #     state = torch.load(model_filename)
    #     model.load_state_dict(state['model_state_dict'])
    #     # initial_epoch = state['epoch'] + 1
    #     optimizer.load_state_dict(state['optimizer_state_dict'])
    #     # global_step = state['global_step']
    # else:
    #     print('No model to preload, starting from scratch')

    

    # scaler = torch.amp.GradScaler('cuda')

    for epoch in range(30):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(loader, desc=f"Processing Epoch {epoch:02d}")
        for imgs, captions, tgt_masks, labels in batch_iterator:

            optimizer.zero_grad(set_to_none=True)
            
            imgs = imgs.to(device)
            
            captions = captions.to(device)
            
            tgt_masks = tgt_masks.to(device)
            
            labels = labels.to(device)
            

            
            

            
                
            output = model(imgs,None,captions,tgt_masks)
                
            loss = loss_fn(output.view(-1, vocab_size), labels.view(-1))
            
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            
            loss.backward()
            
            optimizer.step()


            

        print_examples(model,device,dataset)

        model_filename = "saved_model.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            # 'global_step': global_step
        }, model_filename)