import torch
import torch.nn as nn
import torch.nn.functional as F

class BaselineFusion(nn.Module):
    def __init__(self, img_dim=768, txt_dim=512, hidden_dim=1024, out_dim=768):
        """
        Mô hình Tối giản: Không sử dụng Attention.
        Chỉ lấy CLS Token của Ảnh và EOS Token của Text để nối (Concat) lại.
        """
        super(BaselineFusion, self).__init__()
        
        # Mạng Multi-Layer Perceptron (MLP) để trộn đặc trưng
        self.mlp = nn.Sequential(
            nn.Linear(img_dim + txt_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, img_cls, txt_eos):
        """
        img_cls: [batch_size, 768]
        txt_eos: [batch_size, 512]
        """
        # 1. Chuẩn hóa L2 trước khi gộp (Rất quan trọng cho không gian Cosine)
        img_cls = F.normalize(img_cls, p=2, dim=-1)
        txt_eos = F.normalize(txt_eos, p=2, dim=-1)
        
        # 2. Kết hợp (Concat)
        concat_feat = torch.cat([img_cls, txt_eos], dim=-1) # [batch_size, 768 + 512 = 1280]
        
        # 3. Trộn bằng MLP
        combined_query = self.mlp(concat_feat) # [batch_size, 768]
        
        return combined_query

class AdditiveAttention(nn.Module):
    def __init__(self, query_dim=512, key_dim=768, hidden_dim=512):
        super().__init__()
        self.W_q = nn.Linear(query_dim, hidden_dim)
        self.W_k = nn.Linear(key_dim, hidden_dim)
        self.w_v = nn.Linear(hidden_dim, 1)

    def forward(self, query, keys, values):
        # query: [B, query_dim]
        # keys/values: [B, N, key_dim]
        
        q = self.W_q(query).unsqueeze(1) # [B, 1, hidden_dim]
        k = self.W_k(keys)               # [B, N, hidden_dim]
        
        energy = self.w_v(torch.tanh(q + k)) # [B, N, 1]
        attn_weights = torch.softmax(energy, dim=1) # [B, N, 1] (Softmax qua N mảnh)
        
        context = torch.sum(attn_weights * values, dim=1) # [B, val_dim]
        return context, attn_weights
