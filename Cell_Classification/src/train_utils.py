class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, logits=True, reduce=True):
        """
        Initializes the Focal Loss function.

        Args:
            alpha (float, optional): Weighting factor for the rare class. Defaults to 0.25.
            gamma (float, optional): Focusing parameter for modulating factor (1-p). Defaults to 2.
            logits (bool, optional): Whether inputs are logits. Defaults to True.
            reduce (bool, optional): Whether to reduce the loss. Defaults to True.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        """
        Forward pass for Focal Loss.

        Args:
            inputs (torch.Tensor): Predicted logits.
            targets (torch.Tensor): Ground truth labels.

        Returns:
            torch.Tensor: Computed Focal Loss.
        """
        targets = targets.type_as(inputs)
        if self.logits:
            BCE_loss = nn.functional.binary_cross_entropy_with_logits(
                inputs, targets, reduction="none"
            )
        else:
            BCE_loss = nn.functional.binary_cross_entropy(
                inputs, targets, reduction="none"
            )
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss