import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning.losses import NTXentLoss

class CIRLoss(nn.Module):
    def __init__(self, temperature=0.1):
        """
        Wrapper cho hàm NTXentLoss (InfoNCE).
        Sử dụng nhiệt độ (temperature) để điều chỉnh mức độ phạt các Hard Negatives.
        """
        super(CIRLoss, self).__init__()
        self.loss_fn = NTXentLoss(temperature=temperature)

    def forward(self, combined_query, tgt_imgs_batched):
        """
        combined_query: [batch_size, 768] (Từ mô hình Fusion)
        tgt_imgs_batched: [batch_size, 768] (Ảnh đích gốc)
        """
        batch_size = combined_query.size(0)
        
        # 1. Rút gọn Target Image (Chỉ lấy CLS Token)
        target_cls = tgt_imgs_batched # [batch_size, 768]
        
        # 2. L2 Normalize (QUAN TRỌNG: NTXentLoss tính cosine similarity nên cần normalize)
        query_norm = F.normalize(combined_query, p=2, dim=-1)
        target_norm = F.normalize(target_cls, p=2, dim=-1)
        
        # 3. Nối (Concat) mảng features để đẩy vào thư viện
        # pytorch-metric-learning nhận đầu vào là 1 tensor tổng chứa cả query và target
        # và 1 mảng labels tương ứng để biết ai là 1 cặp.
        embeddings = torch.cat([query_norm, target_norm], dim=0) # [2 * batch_size, 768]
        
        # 4. Tạo Labels
        # Query thứ i và Target thứ i sẽ dùng chung nhãn 'i'.
        labels = torch.arange(batch_size, device=combined_query.device)
        labels = torch.cat([labels, labels], dim=0) # [2 * batch_size]
        
        # 5. Tính Loss
        loss = self.loss_fn(embeddings, labels)
        
        return loss
