"""Loss = CE + lambda_c * BCE(concept, bo qua nhan 9) + lambda_r * Rule.
Rule (bat buoc trong ontology): C16=1 (la khoe, khong ton thuong) -> cac concept ton thuong = 0.
Lukasiewicz: vi pham = relu(p_C16 + p_k - 1), lay trung binh tren 13 concept ton thuong.
P01-P04 KHONG dua vao loss (chi dung de danh gia/giai thich)."""
import torch
import torch.nn.functional as F


class Criterion:
    def __init__(self, cfg, mode, pos_weight=None, device="cpu"):
        self.mode = mode
        self.lambda_c = cfg["cbm"]["lambda_c"]
        self.lambda_r = cfg["cbm"]["lambda_r"]
        if mode == "cbm":
            names = cfg["concepts"]
            self.i_abs = names.index(cfg["absence_concept"])
            self.i_lesion = torch.tensor([i for i, n in enumerate(names) if n != cfg["absence_concept"]], device=device)
            self.pos_weight = pos_weight.to(device)

    def __call__(self, out, y, c):
        """Tra ve (loss tung mau [B], dict tong cac thanh phan de ghi log)."""
        ce = F.cross_entropy(out["class_logits"].float(), y, reduction="none")
        if self.mode != "cbm":
            return ce, {"ce": ce.detach().sum().item(), "loss": ce.detach().sum().item()}
        logits = out["concept_logits"].float()
        mask = (c != 9).float()
        target = torch.where(c == 9, torch.zeros_like(c), c)
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=self.pos_weight, reduction="none") * mask
        bce = bce.sum(1) / mask.sum(1).clamp(min=1.0)
        p = torch.sigmoid(logits)
        rule = F.relu(p[:, self.i_abs:self.i_abs + 1] + p[:, self.i_lesion] - 1.0).mean(1)
        total = ce + self.lambda_c * bce + self.lambda_r * rule
        d = lambda t: t.detach().sum().item()
        return total, {"ce": d(ce), "bce": d(bce), "rule": d(rule), "loss": d(total)}
