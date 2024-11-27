from wireless_tpp.config_factory import Config
from wireless_tpp.runner import TPPRunnerPacketArrival

def train_model(args):

    config = Config.build_from_yaml_file(args.config, experiment_id=args.id)
    model_runner = TPPRunnerPacketArrival(config)
    model_runner.run()
