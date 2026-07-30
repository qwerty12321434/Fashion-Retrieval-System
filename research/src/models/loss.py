import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning.losses import NTXentLoss, TripletMarginLoss

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


class CIRTripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        """
        Wrapper cho hàm TripletMarginLoss.
        Sử dụng margin để điều chỉnh khoảng cách tối thiểu giữa positive và negative pairs.
        """
        super().__init__()
        self.loss_fn = TripletMarginLoss(margin=margin)

    def forward(self, combined_query, tgt_imgs_batched):
        """
        combined_query: [batch_size, 768] (Từ mô hình Fusion)
        tgt_imgs_batched: [batch_size, 768] (Ảnh đích gốc)
        """
        batch_size = combined_query.size(0)
        query_norm = F.normalize(combined_query, p=2, dim=-1)
        target_norm = F.normalize(tgt_imgs_batched, p=2, dim=-1)
        embeddings = torch.cat([query_norm, target_norm], dim=0)
        labels = torch.arange(batch_size, device=combined_query.device).repeat(2)
        return self.loss_fn(embeddings, labels)


class CIRBatchClassificationLoss(nn.Module):
    """
    Batch-based classification loss used by AACL.

    For a batch of B composed queries and B target images, the target at the
    same batch index is the positive class. All other target images in the
    batch are negatives. The paper-faithful configuration is a one-way
    query-to-target dot-product loss without normalization or temperature.
    """

    def __init__(self, normalize=False, temperature=None, symmetric=False):
        super().__init__()
        if temperature is not None and temperature <= 0:
            raise ValueError("temperature must be greater than 0")

        self.normalize = normalize
        self.temperature = temperature
        self.symmetric = symmetric

    def forward(self, combined_query, tgt_imgs_batched):
        if combined_query.ndim != 2 or tgt_imgs_batched.ndim != 2:
            raise ValueError("query and target tensors must both have shape [B, D]")
        if combined_query.shape != tgt_imgs_batched.shape:
            raise ValueError(
                "query and target tensors must have the same [B, D] shape, "
                f"got {tuple(combined_query.shape)} and {tuple(tgt_imgs_batched.shape)}"
            )
        if combined_query.size(0) < 2:
            raise ValueError("batch classification loss requires batch size >= 2")

        query = combined_query
        target = tgt_imgs_batched
        if self.normalize:
            query = F.normalize(query, p=2, dim=-1)
            target = F.normalize(target, p=2, dim=-1)

        logits = query @ target.T
        if self.temperature is not None:
            logits = logits / self.temperature

        labels = torch.arange(logits.size(0), device=logits.device)
        query_to_target = F.cross_entropy(logits, labels)

        if not self.symmetric:
            return query_to_target

        target_to_query = F.cross_entropy(logits.T, labels)
        return 0.5 * (query_to_target + target_to_query)
