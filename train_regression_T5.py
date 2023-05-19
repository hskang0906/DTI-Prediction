import os
import pandas as pd
import wandb

from utils.utils import *
from utils.attention_flow import *

from module.model import BApredictModel
from module.datamodule import BAPredictDataModule

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


## -- Logger Type -- ##
TENSOR = 0
WANDB = 1
SWEEP = 2


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

        log_path = "./log"
        model_type = str(config.pretrained['chem'])+"To"+str(config.pretrained['prot'])
        version_name = f"{config.task_name}_{model_type}_{config.num_seed}"

        if run_type == TENSOR:
            model_logger = TensorBoardLogger(save_dir=log_path, name=version_name)
        else:
            model_logger = WandbLogger(project=project_name, save_dir=log_path, version=version_name)

        
        dm = BAPredictDataModule(config.num_workers, config.batch_size, config.task_name, 
                                 config.d_model_name, config.p_model_name, config.use_T5Model,
                                 config.prot_maxlength, config.drug_maxlength)
        dm.prepare_data()
        dm.setup()

        checkpoint_callback = ModelCheckpoint(f"{config.task_name}_{model_type}_regression", monitor="valid_MSE", mode="min")
        early_stop_callback = EarlyStopping(monitor="valid_loss", min_delta=0.01, patience=10, verbose=10, mode="min")

        trainer = pl.Trainer(devices=config.gpu_ids,
                             max_epochs=config.max_epoch,
                             precision=16,
                             logger=model_logger,
                             accumulate_grad_batches=2,
                             callbacks=[checkpoint_callback, early_stop_callback],
                             accelerator='gpu', 
                             strategy='dp' 
                             )
        
        if config.model_mode == "train":
            model = BApredictModel(config.lr, config.dropout, config.layer_features,
                                    config.d_model_name, config.p_model_name, config.use_T5Model,
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
    run_type = TENSOR

    if run_type == SWEEP:
        config = load_hparams('config/sweep/config_hparam_regression_T5.json')
        project_name = config["name"]
        sweep_id = wandb.sweep(config, project=project_name)
        wandb.agent(sweep_id, main)
    
    else:
        config = load_hparams('config/config_hparam_regression_T5.json')
        project_name = config["name"]
        main(config)