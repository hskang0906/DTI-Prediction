import os
import glob
import numpy as np
import pandas as pd
import wandb

from utils.utils import *
from utils.attention_flow import *
from utils.emetric import regression_score

from module.model import BApredictModel, deleteEncodingLayers
from module.datamodule import BAPredictDataModule

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from transformers import AutoTokenizer, AutoConfig, RobertaModel, BertModel, AutoModel

## -- Logger Type -- ##
TENSOR = 0
WANDB = 1
SWEEP = 2

class DTIpredictDataset(Dataset):
    def __init__(self, drug_input, prot_input, labels):
        'Initialization'
        self.labels = labels
        self.drug_input = drug_input
        self.prot_input = prot_input

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.labels)

    def __getitem__(self, index):
        'Generates one sample of data'
        d_input_ids = self.drug_input['input_ids'][index]
        d_attention_mask = self.drug_input['attention_mask'][index]
        p_input_ids = self.prot_input['input_ids'][index]
        p_attention_mask = self.prot_input['attention_mask'][index]

        labels = torch.as_tensor(self.labels[index], dtype=torch.float)

        dataset = [d_input_ids, d_attention_mask, p_input_ids, p_attention_mask, labels]
        return dataset


class DTIpredictDataModule(pl.LightningDataModule):
    def __init__(self, task_name, drug_model_name, prot_model_name, num_workers, batch_size, prot_maxLength=545, drug_maxLength=512):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.task_name = task_name

        self.prot_maxLength = prot_maxLength
        self.drug_maxLength = drug_maxLength - 2
        
        self.d_tokenizer = AutoTokenizer.from_pretrained(drug_model_name)
        self.p_tokenizer = AutoTokenizer.from_pretrained(prot_model_name)

        self.df_train = None
        self.df_val = None
        self.df_test = None

        self.load_testData = True
        self.load_dataset = False

    def get_task(self, task_name):
        if task_name.lower() == 'bindingdb':
            return './dataset_kd/BindingDB'
        elif task_name.lower() == 'davis':
            return './dataset_kd/DAVIS'
        elif task_name.lower() == 'merge':
            self.load_testData = False
            return './dataset_kd/MergeDataset'

    def tokenization_dataset(self, df_load:pd.DataFrame):
        df_drug = np.array(df_load['Drug']).tolist()
        df_prot = np.array(df_load['Drug']).tolist()
        df_prot = [' '.join(list(aas)) for aas in df_prot]
        label = np.array(df_load['Y']).tolist()

        drug_obj = self.d_tokenizer(df_drug, padding='max_length', max_length=self.drug_maxLength, truncation=True, return_tensors="pt")
        prot_obj = self.p_tokenizer(df_prot, padding='max_length', max_length=self.prot_maxLength, truncation=True, return_tensors="pt")
        

        return drug_obj, prot_obj, label

    def prepare_data(self):
        # Use this method to do things that might write to disk or that need to be done only from
        # a single process in distributed settings.
        if self.load_dataset is False:
            dataFolder = self.get_task(self.task_name)

            df_train = pd.read_csv(dataFolder + '/train.csv')
            df_valid = pd.read_csv(dataFolder + '/valid.csv')
            df_test = pd.read_csv(dataFolder + '/test.csv')

            ## -- tokenization dataset -- ##
            self.drug_train, self.prot_train, self.train_label= self.tokenization_dataset(df_train)
            self.drug_valid, self.prot_valid, self.valid_label= self.tokenization_dataset(df_valid)
            self.drug_test, self.prot_test, self.test_label= self.tokenization_dataset(df_test)
            
            self.load_dataset = True

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = DTIpredictDataset(self.drug_train, self.prot_train, self.train_label)
            self.valid_dataset = DTIpredictDataset(self.drug_valid, self.prot_valid, self.valid_label)

        if self.load_testData is True:
            self.test_dataset = DTIpredictDataset(self.drug_test, self.prot_test, self.test_label)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)


class DTIpredictModel(pl.LightningModule):
    def __init__(self, lr, dropout, layer_features,
                 drug_model_name, prot_model_name, 
                 d_pretrained:bool=True, p_pretrained:bool=True):
        super().__init__()
        self.lr = lr
        self.criterion = torch.nn.SmoothL1Loss()

        #-- Pretrained Model Setting
        drug_config = AutoConfig.from_pretrained(drug_model_name)
        self.d_model = AutoModel(drug_config) if d_pretrained is False else AutoModel.from_pretrained(drug_model_name, num_labels=2,
                                                                                                        output_hidden_states=True,
                                                                                                        output_attentions=True)
        prot_config = AutoConfig.from_pretrained(prot_model_name)
        self.p_model = AutoModel(prot_config) if p_pretrained is False else AutoModel.from_pretrained(prot_model_name,
                                                                                                        output_hidden_states=True,
                                                                                                        output_attentions=True)
        self.p_model = deleteEncodingLayers(self.p_model, 18)

        # self.model = BiomarkerModel.load_from_checkpoint(DTI_model)

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
        d_outputs = self.d_model(drug_inputs['input_ids'], drug_inputs['attention_mask'])
        p_outputs = self.p_model(prot_inputs['input_ids'], prot_inputs['attention_mask'])

        output = torch.cat((d_outputs.last_hidden_state[:, 0], p_outputs.last_hidden_state[:, 0]), dim=1)
        output = self.decoder(output)    

        return output

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

        y_pred = preds.detach().cpu().numpy()
        y_label = labels.detach().cpu().numpy()

        MSE_score, rm2_score, ci_score = regression_score(y_pred, y_label)
                   
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

        y_pred = preds.detach().cpu().numpy()
        y_label = labels.detach().cpu().numpy()

        MSE_score, rm2_score, ci_score = regression_score(y_pred, y_label)

        print(f'CI_score : {ci_score}')
        print(f'MSE_score : {MSE_score}')
        print(f'rm2_score : {rm2_score}')

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
    

def main(config=None):
    try: 
        ##-- hyper param config file Load --##
        if run_type == TENSOR:
            config = DictX(config)
        else:
            if config is not None:
                wandb.init(config=config, project=project_name)
            else:
                wandb.init(settings=wandb.Settings(console='off'))  

            config = wandb.config

        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        pl.seed_everything(seed=config.num_seed)

        DTImodel_seed = {"biosnap":"4183", "davis":"612", "bindingDB":"8595", "merge":"6962"}
        model_type = str(config.pretrained['chem'])+"To"+str(config.pretrained['prot'])
        
        log_path = "./log"
        version_name = f"{config.task_name}_{model_type}_{config.num_seed}"

        if run_type == TENSOR:
            model_logger = TensorBoardLogger(save_dir=log_path, name=version_name)
        else:
            model_logger = WandbLogger(project=project_name, save_dir=log_path, version=version_name)

        

        dm = BAPredictDataModule(config.num_workers, config.batch_size, config.task_name, 
                                 config.d_model_name, config.p_model_name, 
                                 config.prot_maxlength, config.drug_maxlength)
        dm.prepare_data()
        dm.setup()

        checkpoint_callback = ModelCheckpoint(f"{config.task_name}_{model_type}_regression", monitor="valid_MSE", mode="min")
        early_stop_callback = EarlyStopping(monitor="valid_loss", min_delta=0.01, patience=10, verbose=10, mode="min")

        trainer = pl.Trainer(devices=config.gpu_ids,
                             max_epochs=config.max_epoch,
                             precision=16,
                             logger=model_logger,
                             callbacks=[checkpoint_callback, early_stop_callback],
                             accelerator='gpu', 
                             strategy='dp' 
                             )
        
        # model_file = f"./log/{config.task_name}_{modeltype}_{config.lr}_{DTImodel_seed[config.task_name]}/*.ckpt"
        # model_file = glob.glob(model_file)
        # DTIModel = model_file[0]

        if config.model_mode == "train":
            model = BApredictModel(config.lr, config.dropout, config.layer_features,
                                    config.d_model_name, config.p_model_name, 
                                    config.pretrained['chem'], config.pretrained['prot'])
            model.train()
            trainer.fit(model, datamodule=dm)

            model.eval()
            trainer.test(model, datamodule=dm)

        else:
            testmodel_types = ["TrueToTrue", "TrueToFalse", "FalseToTrue", "FalseToFalse"]
            
            for testmodel_type in testmodel_types:
                model_file = f"./log/{config.task_name}_{testmodel_type}_regression/*.ckpt"
                model = BApredictModel.load_from_checkpoint(model_file)
                
                model.eval()
                trainer.test(model, datamodule=dm)

    except Exception as e:
        print(e)


if __name__ == '__main__':
    run_type = SWEEP
    regression_metric = []

    if run_type == SWEEP:
        config = load_hparams('config/sweep/config_hparam_regression.json')
        project_name = config["name"]
        sweep_id = wandb.sweep(config, project=project_name)
        wandb.agent(sweep_id, main)
    
    else:
        config = load_hparams('config/config_hparam_regression.json')
        project_name = config["name"]
        main(config)

        if config["model_mode"] == "test":
            df = pd.DataFrame(regression_metric)
            result_path = f'./results/metric_result/'
            if not os.path.exists(result_path):
                os.makedirs(result_path)

            df.to_csv(os.path.join(result_path, f'/regression_metric_{config["task_name"]}to{config["testdata_name"]}.csv'), index=None)
            print(df)
