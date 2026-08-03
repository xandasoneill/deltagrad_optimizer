import argparse

import joblib

from deltagrad.viz import load_and_plot_results, plot_mean_time_per_epoch


def main():
    parser = argparse.ArgumentParser(
        description="Load two saved benchmark result .pkl files and generate the comparison figures.")
    parser.add_argument("--baseline-results", required=True,
                         help="Path to the baseline optimizer's results .pkl (e.g. Adam).")
    parser.add_argument("--deltagrad-results", required=True,
                         help="Path to the DeltaGrad variant's results .pkl.")
    args = parser.parse_args()

    results_baseline = joblib.load(args.baseline_results)
    results_deltagrad = joblib.load(args.deltagrad_results)

    load_and_plot_results(results_deltagrad, results_baseline)

    print("Baseline total times:", results_baseline["all_total_times"])
    print("DeltaGrad total times:", results_deltagrad["all_total_times"])
    plot_mean_time_per_epoch(results_baseline["all_timestamps"], results_deltagrad["all_timestamps"])


if __name__ == "__main__":
    main()
