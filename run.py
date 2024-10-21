import os
import pandas as pd

import torch
from torch.nn import functional as F
from transformers import AutoTokenizer

from utils.utils import *

from tqdm import tqdm
from train import BiomarkerModel

def get_biomarker(drug_inputs, prot_inputs):
    output_preds = model(drug_inputs, prot_inputs)
    m = torch.nn.Sigmoid()
    predict = torch.squeeze(m(output_preds)).tolist()

    # output_preds = torch.relu(output_preds)
    # predict = torch.tanh(output_preds)
    # predict = predict.squeeze(dim=1).tolist()

    return predict


def biomarker_prediction(smiles, aas, CYP_type):
    try:
        aas_input = []
        for ass_data in aas:
            aas_input.append(' '.join(list(ass_data)))
    
        d_inputs = tokenizer(smiles, padding='max_length', max_length=64, truncation=True, return_tensors="pt")
        # d_inputs = tokenizer(smiles, truncation=True, return_tensors="pt")
        drug_input_ids = d_inputs['input_ids'].to(device)
        drug_attention_mask = d_inputs['attention_mask'].to(device)
        drug_inputs = {'input_ids': drug_input_ids, 'attention_mask': drug_attention_mask}

        p_inputs = prot_tokenizer(aas_input, padding='max_length', max_length=545, truncation=True, return_tensors="pt")
        # p_inputs = prot_tokenizer(aas_input, truncation=True, return_tensors="pt")
        prot_input_ids = p_inputs['input_ids'].to(device)
        prot_attention_mask = p_inputs['attention_mask'].to(device)
        prot_inputs = {'input_ids': prot_input_ids, 'attention_mask': prot_attention_mask}

        #--예측하는 함수는 여기에서 사용.
        output_predict = get_biomarker(drug_inputs, prot_inputs)
        
        output_list = [{'smiles': smiles[i], 'aas': aas[i], 'predict': output_predict[i], 'CYP_type' : CYP_type[i]} for i in range(0,len(aas))]

        return output_list

    except Exception as e:
        print(e)
        return {'Error_message': e}


def smiles_aas_test():
    #--async를 여러번 호출하는 구조를 어떻게 만드냐?
    #--amino acid sequence를 빈칸으로 넣었을때, csv를 읽어서 넣는 방식으로 구성.

    batch_size = 24
    try:
        datas = []
        biomarker_list = []
        biomarker_datas = []

        smiles_aas = pd.read_csv('./dataset/external_dataset_pair.csv')
        
        ## -- 1 to 1 pair predict check -- ##
        for data in smiles_aas.values:
            biomarker_datas.append([data[0], data[1], data[2]])
            if len(biomarker_datas) == batch_size:
                biomarker_list.append(list(biomarker_datas))
                biomarker_datas.clear()

        if len(biomarker_datas) != 0:
            biomarker_list.append(list(biomarker_datas))
            biomarker_datas.clear()
            
        for biomarker_datas in tqdm(biomarker_list, total=len(biomarker_list)):
            smiles_data, ass_data, CYP_type = zip(*biomarker_datas)
            output_pred = biomarker_prediction(list(smiles_data), list(ass_data), list(CYP_type))
            if len(datas) == 0:
                datas = output_pred
            else:
                datas = datas + output_pred

        ## -- Export result data to csv -- ##
        result_path = "./results"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        df = pd.DataFrame(datas)
        df.to_csv(result_path + '/predictData_DAVIS_test.csv', index=None)

        print(df)
        return datas
        
    except Exception as e:
        print(e)
        return {'Error_message': e}

if __name__ == "__main__":

    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0,'

    device_count = torch.cuda.device_count()
    device_biomarker = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    
    device = torch.device('cpu')

    # ------------
    # model
    # ------------
    d_model_name = "seyonec/PubChem10M_SMILES_BPE_450k"
    p_model_name = "Rostlab/prot_bert_bfd"

    tokenizer = AutoTokenizer.from_pretrained(d_model_name)
    prot_tokenizer = AutoTokenizer.from_pretrained(p_model_name)

    #--biomarker Model
    ##-- hyper param config file Load --##
    config = load_hparams('config/config_hparam.json')
    config = DictX(config)
    # model = BiomarkerModel(d_model_name, p_model_name,
    #                            config.lr, config.dropout, config.layer_features, config.loss_fn, config.layer_limit, config.pretrained['chem'], config.pretrained['prot']).to(device_biomarker)

    model = BiomarkerModel.load_from_checkpoint('./davis_TrueToTrue_5e-06_9095/epoch=36-step=2442.ckpt').to(device_biomarker)

    model.eval()
    model.freeze()
        
    if device_biomarker.type == 'cuda':
        model = torch.nn.DataParallel(model)
                             
    smiles_aas_test()