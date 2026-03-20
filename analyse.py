from visualizations import load_and_plot_results, plot_accuracy_comparison, plot_variance_comparison, plot_learning_curves, plot_accuracy_evolution, plot_mean_time_per_epoch
import joblib

# results_adam=joblib.load("results/results_datanoise/20%_50epochs/bs_512/results_values/Adam_results_batch512_lr0.00031705531640654854.pkl")
# results_dg =joblib.load("results/results_datanoise/20%_50epochs/bs_512/results_values/DeltaGrad_results_batch512_lr0.27887247907205764.pkl")
# load_and_plot_results(results_dg, results_adam)
# print(results_adam["all_total_times"])
# print(results_dg["all_total_times"])
# plot_mean_time_per_epoch(results_adam["all_timestamps"], results_dg["all_timestamps"])


results = joblib.load("results/results_datanoise/20%_50epochs/bs_512/results_values/DeltaGrad_results_batch512_lr0.27887247907205764.pkl")
print(results["epochs"])

params =  joblib.load("best_params_DeltaGrad_fixed_b512_epochs50.pkl")
print(params)