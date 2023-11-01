from transformers import AutoConfig, AutoTokenizer, RobertaModel, BertModel

d_tokenizer = AutoTokenizer.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k")
d_tokenizer.save_pretrained("./offline_data/tokenizers/seyonec/PubChem10M_SMILES_BPE_450k")

p_tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert_bfd")
p_tokenizer.save_pretrained("./offline_data/tokenizers/Rostlab/prot_bert_bfd")


roberta_model = RobertaModel.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k")
roberta_model.save_pretrained("./offline_data/models/seyonec/PubChem10M_SMILES_BPE_450k")

bert_model = BertModel.from_pretrained("Rostlab/prot_bert_bfd")
bert_model.save_pretrained("./offline_data/models/Rostlab/prot_bert_bfd")

drug_config = AutoConfig.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k")
drug_config.save_pretrained("./offline_data/configs/seyonec/PubChem10M_SMILES_BPE_450k")

prot_config = AutoConfig.from_pretrained("Rostlab/prot_bert_bfd")
prot_config.save_pretrained("./offline_data/configs/Rostlab/prot_bert_bfd")

