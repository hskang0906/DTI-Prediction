from tdc.multi_pred import DTI

import pandas as pd
import numpy as np

if __name__ == '__main__':
    bindingDB_data = DTI(name = 'BindingDB_Kd')
    davis_data = DTI(name = 'DAVIS')

    bindingDB_data.harmonize_affinities(mode = 'max_affinity')

    bindingDB_data.convert_to_log(form = 'binding')
    davis_data.convert_to_log(form = 'binding')

    split_bindingDB = bindingDB_data.get_split()
    split_davis = davis_data.get_split()

    dataset_list = ["train", "valid", "test"]
    for dataset_type in dataset_list:
        df_bindingDB = pd.DataFrame(split_bindingDB[dataset_type])
        df_davis = pd.DataFrame(split_davis[dataset_type])

        df_bindingDB.to_csv(f"../dataset_kd/bindingDB_{dataset_type}.csv", index=False)
        df_davis.to_csv(f"../dataset_kd/davis_{dataset_type}.csv", index=False)


    Y_bindingDB = np.array(df_bindingDB.Y)
    Y_davis = np.array(df_davis.Y)

    Y_davis_log = [np.log10(Y_davis)]

    