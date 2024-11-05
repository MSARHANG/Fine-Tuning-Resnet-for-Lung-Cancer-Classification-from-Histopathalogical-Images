import torch 
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, image_size, embed_dim, patch_size, num_patches, dropout, in_channel) -> None:
        super().__init__()
        assert image_size % patch_size == 0, 'image size should be divisible by patch size'

        num_patches = (image_size // patch_size) ** 2
        self.position_token = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim), requires_grad=True)


        
        self.patcher = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channel,
                out_channels=embed_dim,
                kernel_size=patch_size,
                stride=patch_size
                ),
            nn.Flatten(2),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim), requires_grad=True)
        #self.position_token = nn.Parameter(torch.randn(1, num_patches+1, embed_dim), requires_grad=True)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
      
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = self.patcher(x)
        x = x.permute(0,2,1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.position_token
        x = self.dropout(x)
        
        return x
    
    
'''
model = PatchEmbedding(IMAGE_SIZE, EMBED_DIM, PATCH_SIZE, NUM_PATCHES, DROPOUT, IN_CHANNELS).to(device)
x = torch.randn(512, 1, 28, 28).to(device)
print(model(x).shape)
'''