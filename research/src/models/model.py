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


class AACLFusion(nn.Module):
    """
    Tái hiện AACL: Additive Attention Composition Layer.

    Thay vì chỉ dùng CLS token và EOS vector (như BaselineFusion),
    AACLFusion nhận TOÀN BỘ:
      - 50 visual tokens của ảnh candidate [B, 50, 768]
        (1 CLS token + 49 spatial patch tokens)
      - L token của câu lệnh text [B, L, 512]

    5 bước theo đúng AACL paper:
      1. Concat: gộp patch + text token thành chuỗi chung [B, 50+L, H]
      2. Additive Self-Attention: học 1 context vector c từ toàn bộ chuỗi
      3. Context vector c [B, 1, H]
      4. Hadamard product: v = c * H (element-wise, điều biến từng token)
      5. Residual + F(v): o = H + F(v) — compose tín hiệu đã điều biến
    Kết quả: pool phần token ảnh (50 token đầu) đã được text compose → [B, 768]
    """
    def __init__(self, img_dim=768, txt_dim=512, hidden_dim=768):
        super().__init__()
        self.img_proj = nn.Linear(img_dim, hidden_dim)
        self.txt_proj = nn.Linear(txt_dim, hidden_dim)
        self.w_v      = nn.Linear(hidden_dim, 1)
        self.F        = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.out_proj = nn.Linear(hidden_dim, img_dim)  # [B, H] → [B, 768] để khớp gallery CLS

    def forward(self, img_patches, txt_tokens, txt_mask):
        """
        img_patches : [B, 50, 768]   — 1 CLS + 49 spatial patch tokens
        txt_tokens  : [B, L,  512]   — full token sequence của text query (đã pad)
        txt_mask    : [B, L]  bool   — True = real token, False = padding
        Returns     : [B, 768]       — composed query vector
        """
        B = img_patches.size(0)

        H_img = self.img_proj(img_patches)       # [B, 50, H]
        H_txt = self.txt_proj(txt_tokens)        # [B, L,  H]

        # Bước 1: Concat patch + text token
        H = torch.cat([H_img, H_txt], dim=1)    # [B, 50+L, H]

        # Mask: tất cả 50 visual token đều real, text dùng txt_mask
        img_mask  = torch.ones(B, img_patches.size(1),
                               device=H.device, dtype=torch.bool)   # [B, 50]
        full_mask = torch.cat([img_mask, txt_mask], dim=1)          # [B, 50+L]

        # Bước 2: Additive Self-Attention
        energy = self.w_v(torch.tanh(H)).squeeze(-1)                 # [B, 50+L]
        energy = energy.masked_fill(~full_mask, -1e9)
        alpha  = torch.softmax(energy, dim=1).unsqueeze(-1)          # [B, 50+L, 1]

        # Bước 3: Context vector
        c = (alpha * H).sum(dim=1, keepdim=True)                     # [B, 1, H]

        # Bước 4: Hadamard product
        v = c * H                                                     # [B, 50+L, H]

        # Bước 5: Residual compose
        o = H + self.F(v)                                            # [B, 50+L, H]

        # Pool chỉ phần patch token ảnh đã được text compose
        n_img = img_patches.size(1)                                  # 50
        query = o[:, :n_img, :].mean(dim=1)                          # [B, H]

        return self.out_proj(query)                                   # [B, 768]
