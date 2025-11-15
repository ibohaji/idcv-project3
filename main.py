from pathlib import Path
from config.settings import load_settings_from_yaml, load_parameters_from_yaml
from experiments.experiment_tracker import ExperimentTracker, Experiment


def main():
    """Main entry point - loads settings and runs all experiments"""
    settings = load_settings_from_yaml(Path("config/settings.yaml"))
    parameters = load_parameters_from_yaml(Path("config/parameters.yaml"))
    
    tracker = ExperimentTracker(settings)
    experiment = Experiment("segmentation_ablation", settings, parameters, tracker)
    
    experiment.run_all(settings.models, settings.datasets, settings.losses, settings.optimizers)
    
    print(f"\n{'='*60}\nAll experiments completed!\n{'='*60}")


if __name__ == "__main__":
    main()
