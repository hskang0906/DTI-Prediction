import glob
import numpy as np
import pandas as pd
import wandb

from utils.utils import *
from utils.attention_flow import *
from utils.emetric import get_cindex, get_rm2

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from transformers import AutoTokenizer
from train import BiomarkerModel

from sklearn.metrics import mean_squared_error

class DTIpredictDataset(Dataset):
    def __init__(self, list_IDs, labels, df_dti, d_tokenizer, p_tokenizer, prot_maxLength):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.df = df_dti

        self.d_tokenizer = d_tokenizer
        self.p_tokenizer = p_tokenizer

        self.prot_maxLength = prot_maxLength

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        index = self.list_IDs[index]
        drug_data = self.df.iloc[index]['Drug']
        prot_data = self.df.iloc[index]['Target']
        prot_data = ' '.join(list(prot_data))

        d_inputs = self.d_tokenizer(drug_data, padding='max_length', max_length=510, truncation=True, return_tensors="pt")
        p_inputs = self.p_tokenizer(prot_data, padding='max_length', max_length=self.prot_maxLength, truncation=True, return_tensors="pt")

        d_input_ids = d_inputs['input_ids'].squeeze()
        d_attention_mask = d_inputs['attention_mask'].squeeze()
        p_input_ids = p_inputs['input_ids'].squeeze()
        p_attention_mask = p_inputs['attention_mask'].squeeze()

        labels = torch.as_tensor(self.labels[index], dtype=torch.float)

        dataset = [d_input_ids, d_attention_mask, p_input_ids, p_attention_mask, labels]
        return dataset


class DTIpredictDataModule(pl.LightningDataModule):
    def __init__(self, task_name, drug_model_name, prot_model_name, num_workers, batch_size, prot_maxLength=545):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.task_name = task_name

        self.prot_maxLength = prot_maxLength
        
        self.d_tokenizer = AutoTokenizer.from_pretrained(drug_model_name)
        self.p_tokenizer = AutoTokenizer.from_pretrained(prot_model_name)

        self.df_train = None
        self.df_val = None
        self.df_test = None

        self.load_testData = True

        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None

    def get_task(self, task_name):
        if task_name.lower() == 'biosnap':
            return './dataset/BIOSNAP/full_data'
        elif task_name.lower() == 'bindingdb':
            return './dataset/BindingDB'
        elif task_name.lower() == 'davis':
            return './dataset/DAVIS'

    def prepare_data(self):
        # Use this method to do things that might write to disk or that need to be done only from
        # a single process in distributed settings.
        dataFolder = self.get_task(self.task_name)

        self.df_train = pd.read_csv(dataFolder + '/train.csv')
        self.df_val = pd.read_csv(dataFolder + '/val.csv')

        traindata_length = int(len(self.df_train))
        validdata_length = int(len(self.df_val))

        self.df_train = self.df_train[:traindata_length]
        self.df_val = self.df_val[:validdata_length]

        self.df_test = pd.read_csv(dataFolder + '/test.csv')

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = DTIpredictDataset(self.df_train.index.values, self.df_train.Y.values, self.df_train,
                                                  self.d_tokenizer, self.p_tokenizer, self.prot_maxLength)
            self.valid_dataset = DTIpredictDataset(self.df_val.index.values, self.df_val.Y.values, self.df_val,
                                                  self.d_tokenizer, self.p_tokenizer, self.prot_maxLength)

        if self.load_testData is True:
            self.test_dataset = DTIpredictDataset(self.df_test.index.values, self.df_test.Y.values, self.df_test,
                                                self.d_tokenizer, self.p_tokenizer, self.prot_maxLength)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)


class DTIpredictModel(pl.LightningModule):
    def __init__(self, DTI_model, lr, dropout, layer_features):
        super().__init__()
        self.lr = lr
        self.criterion = torch.nn.SmoothL1Loss()

        #-- Pretrained Model Setting
        self.model = BiomarkerModel.load_from_checkpoint(DTI_model)

        #-- Decoder Layer Setting
        layers = []
        firstfeature = self.d_model.config.hidden_size + self.p_model.config.hidden_size
        for feature_idx in range(0, len(layer_features) - 1):
            layers.append(nn.Linear(firstfeature, layer_features[feature_idx]))
            firstfeature = layer_features[feature_idx]
            layers.append(nn.ReLU())  
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    
        layers.append(nn.Linear(firstfeature, layer_features[-1]))
        self.decoder = nn.Sequential(*layers)

        self.save_hyperparameters()

    def forward(self, drug_inputs, prot_inputs):
        outputs = self.model(drug_inputs, prot_inputs)

        return outputs

    def training_step(self, batch, batch_idx):
        drug_inputs = {'input_ids': batch[0], 'attention_mask': batch[1]}
        prot_inputs = {'input_ids': batch[2], 'attention_mask': batch[3]}
        labels = batch[4]

        output = self(drug_inputs, prot_inputs)
        logits = output.squeeze(dim=1)
        
        loss = self.criterion(logits, labels)

        self.log("train_loss", loss)
        return {"loss": loss}


    def validation_step(self, batch, batch_idx):
        drug_inputs = {'input_ids': batch[0], 'attention_mask': batch[1]}
        prot_inputs = {'input_ids': batch[2], 'attention_mask': batch[3]}
        labels = batch[4]
        
        output = self(drug_inputs, prot_inputs)
        logits = output.squeeze(dim=1)

        loss = self.criterion(logits, labels)

        self.log("valid_loss", loss, on_step=False, on_epoch=True, logger=True)
        return {"logits": logits, "labels": labels}

    def validation_step_end(self, outputs):
        return {"logits": outputs['logits'], "labels": outputs['labels']}

    def validation_epoch_end(self, outputs):
        preds = self.convert_outputs_to_preds(outputs)
        labels = torch.as_tensor(torch.cat([output['labels'] for output in outputs], dim=0), dtype=torch.int)

        MSE_score, rm2_score, ci_score = self.regression_score(preds, labels)
                   
        self.log("valid_MSE", MSE_score, on_step=False, on_epoch=True, logger=True)
        self.log("valid_rm2", rm2_score, on_step=False, on_epoch=True, logger=True)
        self.log("valid_CI", ci_score, on_step=False, on_epoch=True, logger=True)

    def test_step(self, batch, batch_idx):
        drug_inputs = {'input_ids': batch[0], 'attention_mask': batch[1]}
        prot_inputs = {'input_ids': batch[2], 'attention_mask': batch[3]}
        labels = batch[4]

        output = self(drug_inputs, prot_inputs)
        logits = output.squeeze(dim=1)

        loss = self.criterion(logits, labels)

        self.log("test_loss", loss, on_step=False, on_epoch=True, logger=True)
        return {"logits": logits, "labels": labels}

    def test_step_end(self, outputs):
        return {"logits": outputs['logits'], "labels": outputs['labels']}

    def test_epoch_end(self, outputs):
        preds = self.convert_outputs_to_preds(outputs)
        labels = torch.as_tensor(torch.cat([output['labels'] for output in outputs], dim=0), dtype=torch.int)

        MSE_score, rm2_score, ci_score = self.regression_score(preds, labels)

        self.log("test_MSE", MSE_score, on_step=False, on_epoch=True, logger=True)
        self.log("test_rm2", rm2_score, on_step=False, on_epoch=True, logger=True)
        self.log("test_CI", ci_score, on_step=False, on_epoch=True, logger=True)

    def configure_optimizers(self):
        param_optimizer = list(self.named_parameters())
        no_decay = ["bias", "gamma", "beta"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay_rate": 0.01
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay_rate": 0.0
            },
        ]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.lr,
        )
        return optimizer

    def convert_outputs_to_preds(self, outputs):
        logits = torch.cat([output['logits'] for output in outputs], dim=0)
        return logits

    def regression_score(self, preds, labels):
        y_pred = preds.detach().cpu().numpy()
        y_label = labels.detach().cpu().numpy()

        ci_score = get_cindex(y_label, y_pred)
        MSE_score = mean_squared_error(y_label, y_pred)
        rm2_score = get_rm2(y_label, y_pred)

        print(f'CI_score : {ci_score}')
        print(f'MSE_score : {MSE_score}')
        print(f'rm2_score : {rm2_score}')

        regression_metric.append({"MSE_score": MSE_score, "rm2_score": rm2_score, "CI_score": ci_score})

        return MSE_score, rm2_score, ci_score


def main_wandb(config=None):
    try:
        if config is not None:
            wandb.init(config=config, project=project_name)
        else:
            wandb.init(settings=wandb.Settings(console='off'))

        config = wandb.config
        pl.seed_everything(seed=config.num_seed)

        DTImodel_types = ["FalseToFalse", "TrueToFalse", "FalseToTrue", "TrueToTrue"]
        DTImodel_seed = {"biosnap":"4183", "davis":"612", "bindingDB":"8595", "merge":"6962"}

        modeltype = DTImodel_types[3]

        dm = DTIpredictDataModule(config.task_name, config.d_model_name, config.p_model_name,
                                 config.num_workers, config.batch_size, config.prot_maxlength)
        dm.prepare_data()
        dm.setup()

        model_logger = WandbLogger(project=f"./log/{project_name}")
        checkpoint_callback = ModelCheckpoint(f"{config.task_name}_{modeltype}_regression", monitor="valid_auroc", mode="max")
    
        trainer = pl.Trainer(gpus=config.gpu_ids,
                             max_epochs=config.max_epoch,
                             precision=16,
                             logger=model_logger,
                             callbacks=[checkpoint_callback],
                             accelerator='dp'
                             )
        
        model_file = f"./log/{config.task_name}_{modeltype}_{config.lr}_{DTImodel_seed[config.task_name]}/*.ckpt"
        model_file = glob.glob(model_file)
        DTIModel = model_file[0]

        if config.model_mode == "train":
            model = DTIpredictModel(DTIModel, config.lr, config.dropout, config.layer_features)
            model.train()
            trainer.fit(model, datamodule=dm)

            model.eval()
            trainer.test(model, datamodule=dm)

        else:
            testmodel_types = ["TrueToTrue", "TrueToFalse", "FalseToTrue", "FalseToFalse"]
            
            for testmodel_type in testmodel_types:
                model_file = f"./log/{config.task_name}_{testmodel_type}_regression/*.ckpt"

                model = BiomarkerModel.load_from_checkpoint(model_file)
                
                model.eval()
                trainer.test(model, datamodule=dm)

    except Exception as e:
        print(e)


def main_default(config):
    try:
        config = DictX(config)
        pl.seed_everything(seed=config.num_seed)
    
        dm = DTIpredictDataModule(config.task_name, config.d_model_name, config.p_model_name,
                                 config.num_workers, config.batch_size, config.prot_maxlength)
        dm.prepare_data()
        dm.setup()

        DTImodel_types = ["FalseToFalse", "TrueToFalse", "FalseToTrue", "TrueToTrue"]
        DTImodel_seed = {"biosnap":"4183", "davis":"612", "bindingDB":"8595", "merge":"6962"}

        modeltype = DTImodel_types[3]

        model_logger = TensorBoardLogger("./log", name=f"{config.task_name}_{modeltype}_{config.num_seed}_regression")
        checkpoint_callback = ModelCheckpoint(f"{config.task_name}_{modeltype}_regression", monitor="valid_auroc", mode="max")
    
        trainer = pl.Trainer(gpus=config.gpu_ids,
                             max_epochs=config.max_epoch,
                             precision=16,
                             logger=model_logger,
                             callbacks=[checkpoint_callback],
                             accelerator='dp'
                             )

        model_file = f"./log/{config.task_name}_{modeltype}_{config.lr}_{DTImodel_seed[config.task_name]}/*.ckpt"
        model_file = glob.glob(model_file)

        DTIModel = model_file[0]

        if config.model_mode == "train":
            model = DTIpredictModel(DTIModel, config.lr, config.dropout, config.layer_features)
            model.train()
            trainer.fit(model, datamodule=dm)

            model.eval()
            trainer.test(model, datamodule=dm)

        else:
            testmodel_types = ["TrueToTrue", "TrueToFalse", "FalseToTrue", "FalseToFalse"]
            
            for testmodel_type in testmodel_types:
                model_file = f"./log/{config.task_name}_{testmodel_type}_regression/*.ckpt"

                model = BiomarkerModel.load_from_checkpoint(model_file)
                
                model.eval()
                trainer.test(model, datamodule=dm)

    except Exception as e:
        print(e)


if __name__ == '__main__':
    using_wandb = False
    regression_metric = []

    if using_wandb == True:
        #-- hyper param config file Load --##
        config = load_hparams('config/config_hparam.json')
        project_name = config["name"]
        main_wandb(config)

        ##-- wandb Sweep Hyper Param Tuning --##
        # config = load_hparams('config/config_sweep_bindingDB.json')
        # project_name = config["name"]
        # sweep_id = wandb.sweep(config, project=project_name)
        # wandb.agent(sweep_id, main_wandb)

    else:
        config = load_hparams('config/config_hparam.json')
        main_default(config)

        # if config["model_mode"] == "test":
        #     df = pd.DataFrame(regression_metric)
        #     df.to_csv(f'./results/metric_result/regression_metric_{config["task_name"]}to{config["testdata_name"]}.csv', index=None)
        #     print(df)
