import torch
import torch.nn as nn
from .swin4d_transformer_ver7 import SwinTransformer4D
from einops import rearrange

class SubjectConditioner(nn.Module):
    def __init__(self, num_subjects, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_subjects, embed_dim)
        self.scale = nn.Parameter(torch.ones(1, embed_dim, 1, 1, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, embed_dim, 1, 1, 1, 1))
    
    def forward(self, x, subject_ids):
        # x: (B, C, H, W, D, T)
        # subject_ids: (B,)
        subj_embed = self.embedding(subject_ids) # (B, C)
        subj_embed = subj_embed.view(subj_embed.shape[0], subj_embed.shape[1], 1, 1, 1, 1)
        
        # AdaIN-like modulation
        return x * (1 + self.scale * subj_embed) + (self.bias * subj_embed)

class NeuralMemory(nn.Module):
    def __init__(self, hidden_dim, memory_dim=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.memory_dim = memory_dim if memory_dim else hidden_dim
        
        # Gated Memory Update Mechanism (GRU-like)
        self.reset_gate = nn.Linear(hidden_dim + self.memory_dim, self.memory_dim)
        self.update_gate = nn.Linear(hidden_dim + self.memory_dim, self.memory_dim)
        self.candidate_memory = nn.Linear(hidden_dim + self.memory_dim, self.memory_dim)
        
        self.norm = nn.LayerNorm(self.memory_dim) 

    def forward(self, x, prev_memory=None):
        # x: (B, T, D) -- Feature sequence from encoder
        # prev_memory: (B, D) -- Long-term memory state
        
        B, T, D = x.shape
        
        if prev_memory is None:
            prev_memory = torch.zeros(B, self.memory_dim, device=x.device, dtype=x.dtype)
            
        outputs = []
        current_memory = prev_memory
        
        for t in range(T):
            step_x = x[:, t, :] # (B, D)
            
            combined = torch.cat([step_x, current_memory], dim=-1) # (B, D+D_mem)
            
            update = torch.sigmoid(self.update_gate(combined))
            reset = torch.sigmoid(self.reset_gate(combined))
            
            combined_reset = torch.cat([step_x, reset * current_memory], dim=-1)
            candidate = torch.tanh(self.candidate_memory(combined_reset))
            
            current_memory = (1 - update) * current_memory + update * candidate
            current_memory = self.norm(current_memory)
            
            outputs.append(current_memory.unsqueeze(1))
            
        return torch.cat(outputs, dim=1) # (B, T, D_mem)

class TitansNeuro(nn.Module):
    def __init__(self, 
                 img_size=[96, 96, 96, 20], 
                 in_chans=1, 
                 embed_dim=24, 
                 window_size=[4, 4, 4, 4], 
                 first_window_size=[2, 2, 2, 2],
                 patch_size=[6, 6, 6, 1], 
                 depths=[2, 2, 6, 2], 
                 num_heads=[3, 6, 12, 24],
                 c_multiplier=2,
                 last_layer_full_MSA=False,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 num_subjects=1000,
                 use_memory=True):
        super().__init__()
        
        self.encoder = SwinTransformer4D(
            img_size=img_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            window_size=window_size,
            first_window_size=first_window_size,
            patch_size=patch_size,
            depths=depths,
            num_heads=num_heads,
            c_multiplier=c_multiplier,
            last_layer_full_MSA=last_layer_full_MSA,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate
        )
        
        self.use_memory = use_memory
        if self.use_memory:
            self.memory = NeuralMemory(hidden_dim=embed_dim * (c_multiplier ** (len(depths) - 1)))
            
        self.subject_conditioner = SubjectConditioner(num_subjects, embed_dim)
        
        if last_layer_full_MSA:
             self.num_features = int(embed_dim * c_multiplier ** (len(depths) - 1))
        else:
             self.num_features = int(embed_dim * c_multiplier ** (len(depths) - 1))
             
        self.reconstruction_head = nn.Linear(self.num_features, self.num_features) # Simple head

    def forward(self, x, subject_ids=None):
        # Encoder: (B, C, D, H, W, T) based on SwinTransformer4D output
        features = self.encoder(x) 
        
        # Apply Subject Conditioning on features
        if subject_ids is not None:
             features = self.subject_conditioner(features, subject_ids)

        # Prepare for Memory/Head (B, T, D)
        # Collapse spatial dims: Global Average Pool over (D, H, W) -> dims 2, 3, 4
        # features: (B, C, D, H, W, T)
        
        # Pool spatial dimensions
        features = features.mean(dim=(2, 3, 4)) # (B, C, T)
        features = features.permute(0, 2, 1) # (B, T, C)

        if self.use_memory:
            # Pass through Memory
            features = self.memory(features)
            
            # Output is (B, T, C) - Temporal sequence of memory states
        
        # Project for reconstruction or downstream
        out = self.reconstruction_head(features)
            
        return out
