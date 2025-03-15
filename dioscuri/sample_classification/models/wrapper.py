from torch import nn

class ModelWithLoss(nn.Module):
    """Add utilitarian functions for module to work with pipeline

    Args:
        model (Module): Base Model without loss
        loss (Module): Base loss function with stat

    Example:

        from dioscuri.model.segmentation import MobileUnet
        from dioscuri.model.loss import CEwithstat
        from dioscuri.model import ModelWithLoss

        model = MobileUnet()
        loss = CEwithstat(nclasses = 2)

        modelwithloss = ModelWithLoss(model = model, loss = loss)
    """

    def __init__(self, model: nn.Module, criterion: nn.Module):
        super().__init__()
        self.model = model
        self.criterion = criterion

    def forward(self, batch):
        outputs = self.model(batch["input"])
        loss, loss_dict = self.criterion(outputs, batch)
        return {
            "logit": outputs["logit"],
            "loss": loss,
            "loss_dict": loss_dict
        }

    def forward_train(self, batch):
        return self.forward(batch)

    def forward_eval(self, batch):
        return self.forward(batch)

    def state_dict(self):
        return self.model.state_dict()

    @classmethod
    def from_cfg(cls, model, criterion, getter):
        model = getter(model)
        criterion = getter(criterion)
        return cls(model, criterion)