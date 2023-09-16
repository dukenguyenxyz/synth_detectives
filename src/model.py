import torch

import lightning.pytorch as pl

from torch import optim, nn, utils
from torch.utils.data import Dataset, DataLoader

from transformers import XLNetTokenizer, XLNetModel, AutoTokenizer, AlbertModel, AutoModel, DebertaV2Model, DebertaV2Tokenizer, ElectraModel, RobertaModel, AlbertTokenizer

import numpy as np

import math

from tqdm import tqdm

from argparse import ArgumentParser

# import pytorch_lightning as pl
import torch
from torchmetrics import Accuracy
from torch import nn
from torch.nn import functional as F
# from torch.optim import Adam
from torch.optim.optimizer import Optimizer

# from sklearn.model_selection import train_test_split

class LogisticRegression(pl.LightningModule):
    """
    Logistic regression model
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        bias: bool = True,
        learning_rate: float = 1e-4,
        optimizer: Optimizer = optim.AdamW,
        l1_strength: float = 0.0,
        l2_strength: float = 0.0,
        **kwargs
    ):
        """
        Args:
            input_dim: number of dimensions of the input (at least 1)
            num_classes: number of class labels (binary: 2, multi-class: >2)
            bias: specifies if a constant or intercept should be fitted (equivalent to fit_intercept in sklearn)
            learning_rate: learning_rate for the optimizer
            optimizer: the optimizer to use (default='Adam')
            l1_strength: L1 regularization strength (default=None)
            l2_strength: L2 regularization strength (default=None)
        """
        super().__init__()
        self.save_hyperparameters()
        self.optimizer = optimizer
        # self.accuracy = Accuracy(task='multiclass', num_classes=2, top_k = 1)

        self.validation_step_outputs = []
        self.test_step_outputs = []

        self.linear = nn.Linear(in_features=self.hparams.input_dim, out_features=self.hparams.num_classes, bias=bias)

    def forward(self, x):
        y_hat = self.linear(x)
        return y_hat

    def training_step(self, batch, batch_idx):
        x, y = batch

        # flatten any input
        x = x.view(x.size(0), -1)

        y_hat = self(x)

        # PyTorch cross_entropy function combines log_softmax and nll_loss in single function
        loss = F.cross_entropy(y_hat, y, reduction='sum')

        # L1 regularizer
        if self.hparams.l1_strength > 0:
            l1_reg = sum(param.abs().sum() for param in self.parameters())
            loss += self.hparams.l1_strength * l1_reg

        # L2 regularizer
        if self.hparams.l2_strength > 0:
            l2_reg = sum(param.pow(2).sum() for param in self.parameters())
            loss += self.hparams.l2_strength * l2_reg

        loss /= x.size(0)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self(x)
        # acc = self.accuracy(y_hat, y)
        loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", loss)

    # def on_validation_epoch_end(self):
    #     acc = torch.stack([x['acc'] for x in self.validation_step_outputs]).mean()
    #     val_loss = torch.stack([x['val_loss'] for x in self.validation_step_outputs]).mean()
    #     tensorboard_logs = {'val_ce_loss': val_loss, 'val_acc': acc}
    #     progress_bar_metrics = tensorboard_logs
    #     return {'val_loss': val_loss, 'log': tensorboard_logs, 'progress_bar': progress_bar_metrics}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = torch.argmax(y, dim=1)
        y_hat = torch.argmax(self(x), dim=1)
        accuracy = torch.sum(y == y_hat).item() / (len(y) * 1.0)
        self.log('test_acc', accuracy)

    # def on_test_epoch_end(self):
    #     acc = torch.stack([x['acc'] for x in self.test_step_outputs]).mean()
    #     test_loss = torch.stack([x['test_loss'] for x in self.test_step_outputs]).mean()
    #     tensorboard_logs = {'test_ce_loss': test_loss, 'test_acc': acc}
    #     progress_bar_metrics = tensorboard_logs
    #     return {'test_loss': test_loss, 'log': tensorboard_logs, 'progress_bar': progress_bar_metrics}

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr = 0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        parser.add_argument('--input_dim', type=int, default=None)
        parser.add_argument('--num_classes', type=int, default=None)
        parser.add_argument('--bias', default='store_true')
        parser.add_argument('--batch_size', type=int, default=16)
        return parser

class SoftMaxLit(pl.LightningModule):
    """
    Reference
    https://machinelearningmastery.com/introduction-to-softmax-classifier-in-pytorch/
    """
    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        self.linear = torch.nn.Linear(n_inputs, n_outputs)
        self.softmax = nn.Softmax(dim=1)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.softmax(self.linear(x))
        
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        
        y_hat = self(x)

        loss = self.criterion(y_hat, y)
        
        self.log('train_loss', loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr = 0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
        return optimizer

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss)
        
    def test_step(self, batch, batch_idx):
        x, y = batch
        y = torch.argmax(y, dim=1)
        y_hat = torch.argmax(self(x), dim=1)
        accuracy = torch.sum(y == y_hat).item() / (len(y) * 1.0)
        self.log('test_acc', accuracy)

class Data(Dataset):
    "The data for multi-class classification"
    def __init__(self, df, *, x=None, load_batch_size=None, tokenizer=None, pretrained=None, one_hot=True):
        if one_hot:
            self.y, self.len = self._get_y_and_len_from_df(df)
        else:
            self.y = df['label'].tolist()
            self.len = df['label'].shape[0]
        
        if x is not None:
            self.x = x
        else:
            self.x = self._get_x_from_df(df, load_batch_size, tokenizer, pretrained)
        
    def _get_x_from_df(self, df, load_batch_size, tokenizer, pretrained):
        docs = df['text'].tolist()
        inputs = tokenizer(docs, return_tensors="pt", padding=True)

        cls_arr = []
        for i, (x, y) in zip(tqdm(range(math.ceil(len(df) / load_batch_size))), self._get_x_y_from_df_with_batch(df, load_batch_size)):
            cls = pretrained(**{k: inputs[k][x:y] for k in list(inputs.keys())}).last_hidden_state[:, 0, :].detach()
#             cls = pretrained(**{'input_ids':inputs['input_ids'][x:y],'token_type_ids':inputs['token_type_ids'][x:y],'attention_mask':inputs['attention_mask'][x:y]}).last_hidden_state[:, 0, :].detach()
            cls_arr.append(cls)
        return torch.concat(cls_arr, dtype=torch.float32)
    
    def _get_y_and_len_from_df(self, df):
        dim_0 = df['text'].shape[0]
        matrix = np.zeros((dim_0,2))
        for i, y in enumerate(df['label'].tolist()):
            matrix[i][y] = 1
        return torch.from_numpy(matrix).type(torch.float32), dim_0

    def _get_x_y_from_df_with_batch(self, df, step_size):
        l = list(range(0, len(df), step_size))
        for ind, _ in enumerate(l):
            if l[ind] + step_size >= len(df):
                yield (l[ind], len(df))
            else:    
                yield (l[ind], l[ind + 1])

    def __getitem__(self, idx):
        "accessing one element in the dataset by index"
        return self.x[idx], self.y[idx] 
 
    def __len__(self):
        "size of the entire dataset"
        return self.len

    @staticmethod
    def concat(df, datasets):
        "concatenate dataset embeddings from x provided they are applied on the same df"
        x = torch.cat([dataset.x for dataset in datasets], 1)
        return Data(df, x=x)

# MODELS
class TransformerModel():
    # # XLNet: https://huggingface.co/docs/transformers/model_doc/xlnet # size = 768
    # # ALBERT: https://huggingface.co/docs/transformers/model_doc/albert # size = 768
    # # ELECTRA: 256
    # # Roberta: 768

    MODELS = {
        'albert': {'name': 'albert-base-v2', 'dim': 768,'tokenizer': AlbertTokenizer, 'pretrained': AlbertModel},
        'electra': {'name': 'google/electra-small-discriminator', 'dim': 256,'tokenizer': AutoTokenizer, 'pretrained': ElectraModel},
        'roberta': {'name': 'roberta-base', 'dim': 768,'tokenizer': AutoTokenizer, 'pretrained': RobertaModel},
        'xlnet': {'name': 'xlnet-base-cased', 'dim': 768, 'tokenizer': XLNetTokenizer, 'pretrained': XLNetModel}, 
    }

    def __init__(self, model_tag):
        if model_tag not in list(self.MODELS.keys()):
            raise ValueError(f'Invalid model: {model_tag}. Valide models are: {self.MODELS.join(" ")}')
        
        self.model_tag = model_tag
        self.dim = self.MODELS[model_tag]['dim']
        self.tokenizer = self.MODELS[model_tag]['tokenizer'].from_pretrained(self.MODELS[model_tag]['name'])
        self.pretrained = self.MODELS[model_tag]['pretrained'].from_pretrained(self.MODELS[model_tag]['name'])
        
    def dataset(self, df, dev, save=False):
        # cur_df = df[:100] if dev else df
        dataset = Data(df, load_batch_size = 30, tokenizer=self.tokenizer, pretrained=self.pretrained)  # 10 > 30 > 40 yes # 4 is the best

        if save:
            torch.save(dataset.x, f"pretrained--dev={dev}--model={self.model_tag}.pt")

        return dataset

def get_dataloaders(dataset, batch_size):
    train_dataset, val_dataset, test_dataset = utils.data.random_split(dataset,(0.8, 0.1, 0.1))
    train_dataloader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset = val_dataset, batch_size = batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle=True)
    return {'train': train_dataloader, 'val': val_dataloader, 'test': test_dataloader}