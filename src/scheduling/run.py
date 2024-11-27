from .preprocess import plot_data, create_training_dataset
from .train import train_model
from .predict import generate_predictions, plot_predictions
from .evaluate import evaluate_model

def run_scheduling(inp_args):
    if inp_args.function == "plot_data":
        plot_data(inp_args)
    elif inp_args.function == "create_training_dataset":
        create_training_dataset(inp_args)
    elif inp_args.function == "train_model":
        train_model(inp_args)
    elif inp_args.function == "generate_predictions":
        generate_predictions(inp_args)
    elif inp_args.function == "plot_predictions":
        plot_predictions(inp_args)
    elif inp_args.function == "evaluate":
        evaluate_model(inp_args)
    else:
        print("Invalid function specified")
