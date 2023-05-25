import copy

import torch
import torch.nn as nn

import pytorch_lightning as pl
from transformers import AutoConfig, AutoModel, T5EncoderModel, RobertaModel, BertModel, ElectraModel

from utils.emetric import regression_score


def deleteEncodingLayers(model, num_layers_to_keep):  # must pass in the full bert model
    oldModuleList = model.encoder.layer
    newModuleList = nn.ModuleList()

    # Now iterate over all layers, only keepign only the relevant layers.
    for i in range(num_layers_to_keep):
        newModuleList.append(oldModuleList[i])

    # create a copy of the model, modify it with the new list, and return
    copyOfModel = copy.deepcopy(model)
    copyOfModel.encoder.layer = newModuleList

    return copyOfModel


class BApredictModel(pl.LightningModule):
    def __init__(self, lr, dropout, layer_features,
                 drug_model_name, prot_model_name, use_T5Encoder:bool = False, 
                 d_pretrained:bool=True, p_pretrained:bool=True):
        super().__init__()
        self.lr = lr
        self.criterion = torch.nn.SmoothL1Loss()

        #-- Pretrained Model Setting
        drug_config = AutoConfig.from_pretrained(drug_model_name)
        self.d_model = AutoModel.from_config(drug_config) if d_pretrained is False else AutoModel.from_pretrained(drug_model_name, num_labels=2,
                                                                                                        output_hidden_states=True,
                                                                                                        output_attentions=True)
        prot_config = AutoConfig.from_pretrained(prot_model_name)
        if use_T5Encoder:
            self.p_model = T5EncoderModel(prot_config) if p_pretrained is False else T5EncoderModel.from_pretrained(prot_model_name,
                                                                                                            output_hidden_states=True,
                                                                                                            output_attentions=True)
        else:
            self.p_model = AutoModel.from_config(prot_config) if p_pretrained is False else AutoModel.from_pretrained(prot_model_name,
                                                                                                            output_hidden_states=True,
                                                                                                            output_attentions=True)
            self.p_model = deleteEncodingLayers(self.p_model, 12)

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