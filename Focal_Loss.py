#v1.0
import torch
from torch import nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self,
                 alpha = None,
                 gamma = 2,
                 reduction= 'mean'):
        """Constructor.

        Args:
            alpha (float) [C]: class weights. Defaults to 1.
            gamma (float): constant (the higher, the more important are hard examples). Defaults to 2.
            reduction (str): 'mean', 'sum'. Defaults to 'mean'.
        """
        super().__init__()
        self.alpha = alpha if alpha is not None else 1.0
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, labels):  
        """
        Args:
            logits (_type_): [batch_size, num_classes]
            labels (_type_): [batch_size]

        Returns:
            reduced loss 
        """
        log_p = F.log_softmax(logits, dim=1) #?log_softmax for numerical stability
        p = torch.exp(log_p)
 
        num_classes = logits.size(1)
        labels_one_hot = F.one_hot(labels, num_classes=num_classes).float()
        
        #?select correct class probabilities and alpha, by multiplying for labels_one_hot
        p_t = (p * labels_one_hot).sum(dim=1)
        log_p_t = (log_p * labels_one_hot).sum(dim=1)
        alpha_t = self.alpha if isinstance(self.alpha, torch.Tensor) else torch.tensor(self.alpha).to(logits.device)
        alpha_t = (self.alpha * labels_one_hot).sum(dim=1)

        loss = - alpha_t * ((1 - p_t)**self.gamma) * log_p_t
        
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'none': 
            pass
        return loss