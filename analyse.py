from visualizations import load_and_plot_results, plot_accuracy_comparison, plot_variance_comparison, plot_learning_curves, plot_accuracy_evolution, plot_mean_time_per_epoch
import joblib

# results_adam=joblib.load("results/results_datanoise/20%_50epochs/results_values/Adam_results_batch512_lr0.0003425316690148133.pkl")
# results_dg =joblib.load("results/results_datanoise/20%_50epochs/results_values/DeltaGrad_results_batch512_lr0.10088051249528875.pkl")
# load_and_plot_results(results_dg, results_adam)
# print(results_adam["all_total_times"])
# print(results_dg["all_total_times"])
# plot_mean_time_per_epoch(results_adam["all_timestamps"], results_dg["all_timestamps"])

results =  joblib.load("results/results_datanoise/20%_15epochs/results_values/Adam_results_batch64_lr0.0003425316690148133.pkl")
print(results["optimizer_hyperparameters"])

params = joblib.load("best_params_Adam_fixed_b512_epochs50.pkl")
print(params)