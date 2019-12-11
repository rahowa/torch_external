import torch
import torch.nn as nn 
import torch.nn.functional


class MultiDice(nn.Module):
    """
    Calculate Dice with averaging per classes and 
    then per batch 
    """
    def __init__(self,):
        super(MultiDice, self).__init__()

    def forward(self, outputs, targets):
        smooth = 1e-15
        prediction = outputs.softmax(dim=1)
        dices = []
        
        for val in range(1, 8):
            target = (targets == val).float().squeeze()
            ch_pred = prediction[:, val]
            intersection = torch.sum(ch_pred * target, dim=(1,2))
            union = torch.sum(ch_pred, dim=(1, 2)) + torch.sum(target, dim=(1, 2))      
            dice_part = (2 * intersection + smooth) / (union + smooth)
            dices.append(dice_part.mean())
        return torch.mean(torch.tensor(dices))


class DicePLusCE(nn.Module):
    def __init__(self, dice_coeff):
        super().__init__()
        self.dice_coeff = dice_coeff
        self.ce = nn.CrossEntropyLoss()
        self.dice = MultiDice()

    def forward(self, predicted, targets):
        ce_ = self.ce(predicted, targets)
        dice_ = -torch.log(self.dice(predicted, targets))
        return ce_ + self.dice_coeff * dice_


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss