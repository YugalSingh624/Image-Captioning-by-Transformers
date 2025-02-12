import torch
import torch.nn as nn
import torch.nn.functional as F
import einops


#HyperParameters :
#seting hyperparameters:

device = 'cuda' if torch.cuda.is_available() else 'cpu'

patch_size = 16
latent_size = 512
n_channels = 3
num_heads = 12
num_layers = 12
dropout = 0.1
num_classes = 10
size = 224

epoch = 30
base_lr = 10e-3
weight_decay = 0.03
batch_size = 8





#Multi Head Attention Mechanism:

class Head_normal(nn.Module):
    # One head of self-attention (without masking)
    
    def __init__(self,latent_size = latent_size,dropout = dropout,num_heads = num_heads ):
        super().__init__()

        head_size = latent_size // num_heads
        self.key_normal = nn.Linear(latent_size, head_size, bias=False)
        self.query_normal = nn.Linear(latent_size, head_size, bias=False)
        self.value_normal = nn.Linear(latent_size, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, xk,xq,xv):

        # print('     Entered Head Normal\n')
        
        k = self.key_normal(xk)  # (B, T, head_size)
        q = self.query_normal(xq)   # (B, T, head_size)

        # Compute attention scores (affinities)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # No masking applied
        wei = self.dropout(wei)
        # print(wei.shape)
        # Perform weighted aggregation of values
        v = self.value_normal(xv)  # (B, T, head_size)
        # print(v.shape)
        out = wei @ v  # (B, T, head_size)
        # print('     Exited Head Normal\n')
        return out 

class MultiHeadAttention(nn.Module):
    # Multiple heads of self-attention in parallel (Unmasked)
    
    def __init__(self,latent_size = latent_size, num_head = num_heads, dropout = dropout):
        super().__init__()
        head_size = latent_size // num_head
        self.heads = nn.ModuleList([Head_normal() for _ in range(num_head)])  # Multiple heads
        self.proj = nn.Linear(head_size * num_head, latent_size)  # Projection layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, xk,xq,xv):
        # print(' Entered MultiHead\n')
        out = torch.cat([h(xk,xq,xv) for h in self.heads], dim=-1)  # Concatenate outputs from all heads
        out = self.dropout(self.proj(out))  # Apply projection and dropout
        # print(' Exited Multihead\n')
        return out
    

#Input Embedding for Vision Transformer:
# 1. Create a class which subclasses nn.Module
class InputEmbedding(nn.Module):
    """Turns a 2D input image into a 1D sequence learnable embedding vector.

    Args:
        in_channels (int): Number of color channels for the input images. Defaults to 3.
        patch_size (int): Size of patches to convert input image into. Defaults to 16.
        embedding_dim (int): Size of embedding to turn image into. Defaults to 768.
    """
    # 2. Initialize the class with appropriate variables
    def __init__(self,patch_size =patch_size,n_channels =n_channels,device= device,latent_size = latent_size,batch_size= batch_size):
        super().__init__()


        self.patch_size = patch_size
        self.latent_size =latent_size
        self.n_channels = n_channels
        self.device = device
        self.batch_size = batch_size
        self.input_size = self.patch_size *self.patch_size * self.n_channels

        # 3. Create a layer to turn an image into patches
        self.patcher = nn.Conv2d(in_channels=3,
                                 out_channels=self.latent_size,
                                 kernel_size=16,
                                 stride=16,
                                 padding=0)

        # 4. Create a layer to flatten the patch feature maps into a single dimension
        self.flatten = nn.Flatten(start_dim=2, # only flatten the feature map dimensions into a single vector
                                  end_dim=3)
        

        # class token
        self.class_token = nn.Parameter(torch.randn(self.batch_size,1,self.latent_size)).to(self.device)
        # print(self.class_token.shape)
        # position embedding
        self.pos_embedding = nn.Parameter(torch.randn(self.batch_size,1,self.latent_size)).to(self.device)

    # 5. Define the forward method
    def forward(self, x):
        # Create assertion to check that inputs are the correct shape
        image_resolution = x.shape[-1]
        assert image_resolution % patch_size == 0, f"Input image size must be divisible by patch size, image shape: {image_resolution}, patch size: {patch_size}"

        # Perform the forward pass
        x_patched = self.patcher(x)
        # print("X_patches size:",x_patched.shape)
        x_flattened = self.flatten(x_patched)

        # print("x_flatted normal:", x_flattened.shape)

        x_flattened = x_flattened.permute(0, 2, 1).to(self.device)
        b , n, _ = x_flattened.shape

        # print("x_flattened permuted:", x_flattened.shape)

        liner_projection = torch.cat((self.class_token, x_flattened), dim=1)
        # print(liner_projection.shape)
        pos_embedding = einops.repeat(self.pos_embedding, 'b 1 d -> b m d', m = n+1)

        liner_projection = liner_projection +pos_embedding
        
        return liner_projection# adjust so the embedding is on the final dimension [batch_size, P^2•C, N] -> [batch_size, N, P^2•C]
    

#Encoder Block For ViT:
class EncoderBlock(nn.Module):
    def __init__(self,latent_size = latent_size, num_heads = num_heads, device = device, dropout = dropout):
        super(EncoderBlock,self).__init__()

        self.latent_size = latent_size
        self.num_heads = num_heads
        self.device = device
        self.dropout = dropout

        #nOrm layer

        self.norm = nn.LayerNorm(self.latent_size)

        #MULTIHEADATTENTION

        # self.multihead = nn.MultiheadAttention(
        #     self.latent_size, self.num_heads, self.dropout
        # )

        self.multihead = MultiHeadAttention()

        # nn.MultiheadAttention()

        self.enc_MLP = nn.Sequential(
            nn.Linear(self.latent_size, self.latent_size *4),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.latent_size *4 , self.latent_size),
            nn.Dropout(self.dropout)
            
        )

    
    def forward(self, embedded_patches):

        first_norm = self.norm(embedded_patches)
        attention_out = self.multihead(first_norm,first_norm,first_norm)[0]

        # first_residuloa_connetion

        first_added = attention_out + embedded_patches

        second_norm =self.norm(first_added)

        ff_output = self.enc_MLP(second_norm)


        return ff_output + first_added
    

#puting everything toggether:

class VitModel(nn.Module):
    def __init__(self,num_encoders = num_layers, latent_size = latent_size, device = device,num_classes = num_classes, dropout = dropout):
        super(VitModel,self).__init__()

        self.num_encoders = num_encoders
        self.latent_size =latent_size
        self.device = device
        self.dropout = dropout
        self.num_classes = num_classes

        self.embd = InputEmbedding()

        self.encstack = nn.ModuleList([ EncoderBlock() for i in range(self.num_encoders)])


    def forward(self, test_input):

        enc_output = self.embd(test_input)

        for enc_layer in self.encstack:
            enc_output = enc_layer.forward(enc_output)
        
        

        return enc_output
        

    

# model = VitModel().to(device)
# test_input = torch.randn((32,3,224,224)).to(device)
# print(model(test_input).shape)


# print(sum(p.numel() for p in model.parameters() if p.requires_grad))