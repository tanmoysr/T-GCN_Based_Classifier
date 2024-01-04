import argparse
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
import utils.metrics
import utils.losses
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


class SupervisedForecastTask(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        regressor="logistic", # linear, logistic
        loss="CrossEntropyLoss", # mse, mse_with_regularizer, CrossEntropyLoss
        pre_len: int = 3,
        learning_rate: float = 1e-3,
        weight_decay: float = 1.5e-3,
        feat_max_val: float = 1.0,
        **kwargs
    ):
        super(SupervisedForecastTask, self).__init__()
        self.save_hyperparameters() # Commented by TC
        self.model = model

        # self.ordinal_weight = torch.rand(self.hparams.batch_size, self.hparams.num_class) # added by TC
        if regressor == "linear":
            self.regressor = nn.Linear(
                                     self.model.hyperparameters.get("hidden_dim")
                                     or self.model.hyperparameters.get("output_dim"),
                                     self.hparams.pre_len)
        elif regressor == "logistic": ## Added by TC
            self.regressor = nn.Linear(self.model.hyperparameters.get("hidden_dim")
                                 or self.model.hyperparameters.get("output_dim"),
                                 self.hparams.num_class)
        else:
            self.regressor = regressor
        # self.regressor = (
        #     nn.Linear(
        #         self.model.hyperparameters.get("hidden_dim")
        #         or self.model.hyperparameters.get("output_dim"),
        #         # self.hparams.pre_len
        #         self.hparams.num_class, # added by TC
        #     )
        #     if regressor == "linear"
        #     else regressor
        # )
        self._loss = loss
        self.feat_max_val = feat_max_val

    def forward(self, x):
        # (batch_size, seq_len, num_nodes)
        batch_size, _, num_nodes = x.size()
        # (batch_size, num_nodes, hidden_dim)
        hidden = self.model(x)
        # (batch_size * num_nodes, hidden_dim)
        hidden = hidden.reshape((-1, hidden.size(2)))
        # (batch_size * num_nodes, pre_len)
        if self.regressor is not None:
            predictions = self.regressor(hidden)
        else:
            predictions = hidden
        predictions = predictions.reshape((batch_size, num_nodes, -1))

        # if self.regressor == 'logistic':
        #     return torch.sigmoid(predictions) # added by TC
        #     # return torch.add(torch.ones(predictions.shape[0],predictions.shape[1], requires_grad=True), torch.argmax(F.softmax(predictions), dim=2)).unsqueeze(-1) # added by TC
        #     # return torch.argmax(F.softmax(predictions), dim=2).unsqueeze(-1).double() # added by TC
        # else:
        #     return predictions
        return torch.sigmoid(predictions)  # added by TC

    def shared_step(self, batch, batch_idx):
        # (batch_size, seq_len/pre_len, num_nodes)
        x, y = batch
        num_nodes = x.size(2)
        predictions = self(x)
        # if self.regressor != 'logistic':
        #     predictions = predictions.transpose(1, 2).reshape((-1, num_nodes))
        # predictions = predictions.transpose(1, 2).double() # Modified by TC
        y = y.reshape((-1, y.size(2)))
        return predictions, y

    def loss(self, inputs, targets):
        if self._loss == "mse":
            return F.mse_loss(inputs, targets)
        if self._loss == "CrossEntropyLoss": # added by TC
            CEloss = nn.CrossEntropyLoss() # multi-class regression
            # CEloss = nn.CrossEntropyLoss(weight = torch.diag (self.ordinal_weight)) # ordinal regression
            return CEloss(inputs.reshape((-1, inputs.shape[-1])), targets.ravel().type(torch.LongTensor))
            # return CEloss(inputs, targets)
        if self._loss == "mse_with_regularizer":
            return utils.losses.mse_with_regularizer_loss(inputs, targets, self)
        raise NameError("Loss not supported:", self._loss)

    def training_step(self, batch, batch_idx):
        predictions, y = self.shared_step(batch, batch_idx)
        # predictions = torch.max(predictions.data, 2, keepdim=True)[1].double()  # added by TC if loss=mse
        loss = self.loss(predictions, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        predictions, y = self.shared_step(batch, batch_idx)

        # predictions = torch.add(torch.ones(predictions.shape[0], predictions.shape[1], requires_grad=True),
        #                         torch.argmax(F.softmax(predictions), dim=2)).unsqueeze(-1)  # added by TC



        # predictions = torch.add(torch.ones(predictions.shape[0],predictions.shape[1], requires_grad=True), torch.argmax(F.softmax(predictions), dim=2)).unsqueeze(-1) # added by TC
        # predictions = torch.argmax(F.softmax(predictions), dim=2).unsqueeze(-1).double() # added by TC

        ## Commented by TC
        # predictions = predictions * self.feat_max_val
        # y = y * self.feat_max_val
        # predictions = torch.max(predictions.data, 2)[1].double()  # added by TC if loss=mse
        loss = self.loss(predictions, y)
        # if self.regressor == 'logistic':
        predictions = torch.max(predictions.data,2)[1].double() # added by TC if loss!=mse
        rmse = torch.sqrt(torchmetrics.functional.mean_squared_error(predictions, y))
        mae = torchmetrics.functional.mean_absolute_error(predictions, y)
        accuracy = utils.metrics.accuracy(predictions, y)
        r2 = utils.metrics.r2(predictions, y)
        explained_variance = utils.metrics.explained_variance(predictions, y)
        # precision_metric = torchmetrics.classification.MulticlassPrecision(num_classes=self.hparams.num_class)
        # precision_metric = torchmetrics.classification.MultilabelPrecision(num_labels=self.hparams.num_class)
        # precision = precision_metric(predictions, y)
        precision = precision_score(y.cpu().detach().numpy().flatten().astype(int),
                                    predictions.cpu().detach().numpy().flatten().astype(int),
                                    average = 'macro')
        # recall_metric = torchmetrics.classification.MulticlassRecall(num_classes=self.hparams.num_class)
        # recall_metric = torchmetrics.classification.MultilabelRecall(num_labels=self.hparams.num_class)
        # recall = recall_metric(predictions, y)
        recall = recall_score(y.cpu().detach().numpy().flatten().astype(int),
                              predictions.cpu().detach().numpy().flatten().astype(int),
                              average='macro')
        # f1_metric = torchmetrics.classification.MulticlassF1Score(num_classes=self.hparams.num_class)
        # f1_metric = torchmetrics.classification.MultilabelF1Score(num_labels=self.hparams.num_class)
        # f1_score = f1_metric(predictions, y)
        f1 = f1_score(y.cpu().detach().numpy().flatten().astype(int),
                              predictions.cpu().detach().numpy().flatten().astype(int),
                              average='macro')
        mze = ((predictions != y).long().sum() / y.ravel().shape[0]).item()
        metrics = {
            "val_loss": loss,
            "RMSE": rmse,
            "MZE": mze,
            "MAE": mae,
            "accuracy": accuracy,
            "R2": r2,
            "ExplainedVar": explained_variance,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        }
        self.log_dict(metrics)
        return predictions.reshape(batch[1].size()), y.reshape(batch[1].size())

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

    @staticmethod
    def add_task_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", "--lr", type=float, default=1e-3)
        parser.add_argument("--weight_decay", "--wd", type=float, default=1.5e-3)
        # parser.add_argument("--loss", type=str, default="mse")
        parser.add_argument("--loss", type=str, default="CrossEntropyLoss") # added by TC

        return parser
