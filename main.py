import argparse
from src import preprocess_edaf, run_link_quality, run_packet_arrival, run_scheduling, run_e2e

def main():
    parser = argparse.ArgumentParser(description="Task parser")
    parser.add_argument("-t", "--task", choices=[
            "preprocess_edaf",
            "link_quality",
            "packet_arrival",
            "scheduling",
            "e2e"
        ],
        required=True,
        help="Specify the task to run"
    )

    # Parse the task argument first
    args, remaining_args = parser.parse_known_args()

    if args.task == "preprocess_edaf":
        sub_parser = argparse.ArgumentParser(description="Preprocessing EDAF")
        sub_parser.add_argument("-s", "--source", required=True, help="Specify the source directory")
        final_args = sub_parser.parse_args(remaining_args)
        preprocess_edaf(final_args)
        return 
    
    sub_parser = argparse.ArgumentParser(description="Functions parser")
    sub_parser.add_argument("-u", "--function", choices=[
            "plot_data",
            "create_training_dataset",
            "train_model",
            "generate_predictions",
            "plot_predictions",
            "evaluate_model"
        ],
        required=True,
        help="Specify the function to run"
    )
    sub_parser.add_argument("-s", "--source", help="Specify the source directory")
    sub_parser.add_argument("-v", "--interarrival", action="store_true", help="Specify if in scheduling plot, plot interarrival histograms")
    sub_parser.add_argument("-f", "--fast", action="store_true", help="Specify if in plot_link_data, only priliminary data should be plotted")
    sub_parser.add_argument("-c", "--config", help="Specify the configuration file")
    sub_parser.add_argument("-g", "--configname", help="Specify the configuration name in the configuration file")
    sub_parser.add_argument("-n", "--name", help="Specify the name of the dataset")
    sub_parser.add_argument("-i", "--id", help="Specify the training id")
    sub_parser.add_argument("-m", "--segment", help="Specify the segment number to plot")
    sub_parser.add_argument("-p", "--predict", choices=["probabilistic","sampling"],help="Specify the prediction method")
    sub_parser.add_argument("-k", "--notypemapping", action="store_true", help="Specify if in arrival create dataset, no type mapping should be done")
    final_args = sub_parser.parse_args(remaining_args)

    if args.task == "link_quality":
        run_link_quality(final_args)
    elif args.task == "packet_arrival":
        run_packet_arrival(final_args)
    elif args.task == "scheduling":
        run_scheduling(final_args)
    elif args.task == "e2e":
        run_e2e(final_args)
    else:
        print("Invalid task specified")
        
if __name__ == "__main__":
    main()

