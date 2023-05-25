import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from transformers import AutoTokenizer, T5Tokenizer 


def get_task(task_name):
    if task_name.lower() == 'bindingdb':
        return './dataset_kd/BindingDB'
    elif task_name.lower() == 'davis':
        return './dataset_kd/DAVIS'
    elif task_name.lower() == 'merge':
        return './dataset_kd/MergeDataset'


class BAPredictDataset(Dataset):
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
    

class BAPredictDataModule(pl.LightningDataModule):
    def __init__(self, num_workers, batch_size, task_name, 
                 drug_model_name, prot_model_name, use_T5model:bool = False,
                 prot_maxLength=545, drug_maxLength=512, fix_testFile = None):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.task_name = task_name
        self.use_T5model = use_T5model

        self.prot_maxLength = prot_maxLength
        self.drug_maxLength = drug_maxLength - 2

        self.fix_testFile = fix_testFile
        
        self.d_tokenizer = AutoTokenizer.from_pretrained(drug_model_name)
        if self.use_T5model:
            self.p_tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc", do_lower_case=False)
        else:
            self.p_tokenizer = AutoTokenizer.from_pretrained(prot_model_name)

        self.load_testData = True
        self.load_dataset = False

    def tokenization_dataset(self, df_load:pd.DataFrame):
        df_drug = np.array(df_load['Drug']).tolist()
        df_prot = np.array(df_load['Drug']).tolist()
        df_prot = [' '.join(list(aas)) for aas in df_prot]
        label = np.array(df_load['Y']).tolist()

        drug_obj = self.d_tokenizer(df_drug, padding='max_length', max_length=self.drug_maxLength, truncation=True, return_tensors="pt")
        
        if self.use_T5model:
            # prot_obj = self.p_tokenizer(df_prot, add_special_tokens=True, padding='longest', return_tensors="pt")
            prot_obj = self.p_tokenizer(df_prot, padding='max_length', max_length=self.prot_maxLength, truncation=True, return_tensors="pt")
        else:
            prot_obj = self.p_tokenizer(df_prot, padding='max_length', max_length=self.prot_maxLength, truncation=True, return_tensors="pt")
        
        return drug_obj, prot_obj, label

    def prepare_data(self):
        # Use this method to do things that might write to disk or that need to be done only from
        # a single process in distributed settings.
        if self.load_dataset is False:
            dataFolder = get_task(self.task_name)

            df_train = pd.read_csv(dataFolder + '/train.csv')
            df_valid = pd.read_csv(dataFolder + '/valid.csv')

            if self.fix_testFile is None:
                df_test = pd.read_csv(dataFolder + '/test.csv')
            else:
                df_test = pd.read_csv(self.fix_testFile + '/test.csv')

            ## -- tokenization dataset -- ##
            self.drug_train, self.prot_train, self.train_label= self.tokenization_dataset(df_train)
            self.drug_valid, self.prot_valid, self.valid_label= self.tokenization_dataset(df_valid)
            self.drug_test, self.prot_test, self.test_label= self.tokenization_dataset(df_test)
            
            self.load_dataset = True

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = BAPredictDataset(self.drug_train, self.prot_train, self.train_label)
            self.valid_dataset = BAPredictDataset(self.drug_valid, self.prot_valid, self.valid_label)

        if self.load_testData is True:
            self.test_dataset = BAPredictDataset(self.drug_test, self.prot_test, self.test_label)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
