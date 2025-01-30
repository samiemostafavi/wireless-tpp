from pathlib import Path
from wireless_tpp.config_factory import Config
from wireless_tpp.runner import TPPRunnerE2E

def train_model(args):

    dataset_id = args.name # -n dataset_id
    base_dir = Path(args.source) / "e2e" / "trained_models" / (dataset_id + "_" + args.id)

    config = Config.build_from_yaml_file(args.config, experiment_id=args.id, base_dir=base_dir, dataset_id=dataset_id)
    model_runner = TPPRunnerE2E(config)
    model_runner.run()
