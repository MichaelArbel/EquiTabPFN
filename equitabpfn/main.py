import warnings
import argparse
import yaml
import os


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_dir", "-c", default="configs",
                        help="Path to config dir")


    parser.add_argument("--train_config", "-t", default="train",
                        help="Path to JSON config file")


    parser.add_argument("--model_config", "-m", default="model/equitabpfn",
                        help="Path to JSON config file")


    parser.add_argument("--prior_config", "-p", default="prior/prior",
                        help="Path to JSON config file")


    parser.add_argument("--output_dir", "-o", default="logs",
                        help="Output directory for logs (default: logs/)")

    parser.add_argument("--run_name", "-n", default=None,
                        help="Name of log subdirectory (default: auto timestamp)")
 
    parser.add_argument("--data_path", "-d", default=None,
                        help="Name of directory containing evaluation data")
    
    return parser.parse_args()

def main():


    from equitabpfn.trainer import Trainer
    from equitabpfn.utils import set_seed, ConfigDict, SimpleFSLogger
    import equitabpfn.models.equitabpfn


    args = parse_args()
    # --------------------------
    # Load JSON config
    # --------------------------
    
    config_dir = os.path.join(args.config_dir, args.train_config+".yaml")
    model_dir = os.path.join(args.config_dir, args.model_config+".yaml")
    prior_dir = os.path.join(args.config_dir, args.prior_config+".yaml")

    with open(config_dir, "r") as f:
        config_dict = yaml.safe_load(f)
    with open(model_dir, "r") as f:
        model_dict = yaml.safe_load(f)
    with open(prior_dir, "r") as f:
        prior_dict = yaml.safe_load(f)

    config = ConfigDict(config_dict)
    config.model = model_dict
    config.prior = prior_dict
    config.data_path = args.data_path

    logger = SimpleFSLogger(root_dir = args.output_dir, run_name=args.run_name)

    logger.log_metadata(config)

    set_seed(config.seed)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")

        exp = Trainer(config,logger)
        try:
            if config.load.load_existing_cktp:
                ckpt = logger.load_artifacts(artifact_name="ckpt/last")
                exp.load_checkpoint(ckpt)
                print("Loading from latest checkpoint")
        except:
            print("Failed to load checkpoint, Starting from scratch")

        if config.mode.train_mode:
            exp.train()

if __name__ == "__main__":
    main()
