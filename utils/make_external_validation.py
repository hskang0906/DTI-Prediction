import numpy as np
import pandas as pd


if __name__ == '__main__':
    smiles = pd.read_csv("../dataset/external_smiles.csv")
    ass = pd.read_csv("../dataset/external_aas.csv")
    
    smiles_data = list(np.array(smiles['smiles']))
    smiles_label = list(np.array(smiles['label'].tolist()))
    smiles_label = [x.split() for x in smiles_label]

    ass_data = list(np.array(ass['aas']))
    cyp_type = list(np.array(ass['CYP_type']))

    external_dataset = []
    for smiles_idx in range(0, len(smiles_data)):
        for ass_idx in range(0, len(ass_data)):
            
            external_data = [smiles_data[smiles_idx], ass_data[ass_idx], cyp_type[ass_idx]]
            external_dataset.append(external_data)

    df = pd.DataFrame(external_dataset, columns=['smiles', 'aas', 'CYP_type'])
    df.to_csv('../dataset/external_dataset.csv', index=False)


    print(smiles['smiles'][0])
    print(ass['CYP_type'][0])