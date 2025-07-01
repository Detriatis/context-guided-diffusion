import optuna
import logging
import argparse
from pathlib import Path 
from optuna.integration import BoTorchSampler
from swiss_roll import RUNS_DIR
from swiss_roll.guidance import run_once
from swiss_roll.config import OptParams, CGDParams, TrainParams, RunParams, SamplerParams, Config, load_config, save_config, load_hyperoptconfig

class HyperOptRun: 
    def __init__(self, conf_path: Path, log_path: Path, writeout: Path):
        self.conf_path = conf_path 
        self.log_path = log_path 
        self.writeout = writeout 

        self.cfg =  load_hyperoptconfig(conf_path)
        self.logger = self.init_logger(log_path)
        self.sampler = self.init_sampler()
        self.study = self.init_study()

    def __call__(self):
        self.study.optimize(self.objective, n_trials=self.cfg.trials, timeout=3*60*60)   # 3 h cap
    
    def init_logger(self, log_path: Path):
        logging.basicConfig(
            level=logging.INFO,
            filename = log_path.with_suffix(".log"),
            filemode = "w",
        )

        logger = logging.getLogger("__name__")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        
        return logger 

    def init_study(self) -> optuna.Study:
        self.run_id = f"{int(self.cfg.cfg.run.run_id):02d}"
        if not self.cfg.pareto:
            study = optuna.create_study(
                    direction="minimize",
                    sampler=self.sampler,
                    study_name=f"CGD_SwissRoll_BayesOpt_Single_{self.run_id}",
                )
            self.objective = self.objective_single

        if self.cfg.pareto: 
            study = optuna.create_study(
                directions=["minimize", "minimize"],
                sampler=self.sampler,
                study_name=f"CGD_SwissRoll_BayesOpt_Pareto_{self.run_id}",
            )
            self.objective = self.objective_pareto

        return study 

    def init_sampler(self) -> BoTorchSampler:
        sampler = BoTorchSampler(
                n_startup_trials=self.cfg.startup,        # random warm-up
                seed=self.cfg.seed,                    # reproducibility
            )
        return sampler 

    def writeout_study(self):
        df = self.study.trials_dataframe()
        
        df.to_csv(self.writeout / f"trials_{self.run_id}.csv", index=False)
        save_config(self.cfg, self.writeout / f"conf_{self.run_id}.yaml")


    def objective_single(self, trial: optuna.trial.Trial) -> float:
        cfg = self.cfg.cfg  
        
        #  continuous log-uniform search space
        if cfg.run.reg_type == 'cgd':  
            # cfg.cgd.reg_lambda = trial.suggest_float("reg_lambda", 1e-2, 1e4)
            cfg.cgd.sigma_t    = trial.suggest_float("sigma_t", 1e-5,  1e5, log=True)
            cfg.cgd.tau_t      = trial.suggest_float("tau_t", 1e-5, 1e5, log=True)
        cfg.opt.l2_lambda  = trial.suggest_float("l2_lambda", 1e-4, 1e4, log=True)
        cfg.opt.lr         = trial.suggest_float("lr", 1e-3, 1e-2, log=True)
        
        training_metrics, validation_metrics, guidance_model = run_once(cfg, self.logger)
        loss = validation_metrics.loss
        
        self.logger.info(f"Trial {trial.number}: train NLL={training_metrics.nll:.4f}") 
        self.logger.info(f"Trial {trial.number}: train L2={training_metrics.l2:.4f}") 
        self.logger.info(f"Trial {trial.number}: train MSE={training_metrics.mse:.4f}") 
         
        self.logger.info(f"Trial {trial.number}: val NLL={validation_metrics.nll:.4f}") 
        self.logger.info(f"Trial {trial.number}: val L2={validation_metrics.l2:.4f}") 
        self.logger.info(f"Trial {trial.number}: val MSE={validation_metrics.mse:.4f}") 
        
        return loss
    
    def objective_pareto(self, trial: optuna.trial.Trial):
        cfg = self.cfg.cfg

        if cfg.run.reg_type == 'cgd': 
            cfg.cgd.sigma_t    = trial.suggest_float("sigma_t", 1e-5,  1e5, log=True)
            cfg.cgd.tau_t      = trial.suggest_float("tau_t", 1e-5, 1e5, log=True)
        
        cfg.opt.l2_lambda  = trial.suggest_float("l2_lambda", 1e-4, 1e4, log=True)
        cfg.opt.lr         = trial.suggest_float("lr", 1e-3, 1e-2, log=True)
        
        training_metrics, validation_metrics, guidance_model = run_once(cfg, self.logger)
        train_loss = training_metrics.loss 
        val_loss = validation_metrics.loss
       
        self.logger.info(f"Trial {trial.number}: train NLL={training_metrics.nll:.4f}") 
        self.logger.info(f"Trial {trial.number}: train L2={training_metrics.l2:.4f}") 
        self.logger.info(f"Trial {trial.number}: train MSE={training_metrics.mse:.4f}") 
         
        self.logger.info(f"Trial {trial.number}: val NLL={validation_metrics.nll:.4f}") 
        self.logger.info(f"Trial {trial.number}: val L2={validation_metrics.l2:.4f}") 
        self.logger.info(f"Trial {trial.number}: val MSE={validation_metrics.mse:.4f}") 
       
        return train_loss, val_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", "-c", help='Full Conf Path') 
    parser.add_argument("--writeout", "-w", help="Full Writeout Path", required=False, default=None) 
    parser.add_argument("--logger", "-l", help="Logger Path") 
    args = parser.parse_args()
    if args.writeout is None: 
        args.writeout = '/dev/null'
    run = HyperOptRun(Path(args.conf), Path(args.logger), Path(args.writeout))
    run()
    run.writeout_study() 
    

   
    