from visualizations import load_and_plot_results, plot_accuracy_comparison, plot_variance_comparison, plot_learning_curves, plot_accuracy_evolution, plot_time_per_epoch_comparison
import joblib

results_adam=joblib.load("results/results_batchtest/results_values/Adam_results_batch64_lr0.00017403859418352828.pkl")
results_dg =joblib.load("results/results_batchtest/results_values/DeltaGrad_results_batch64_lr0.03194028510565753.pkl")
load_and_plot_results(results_dg, results_adam)
print(results_adam["total_time"])
print(results_dg["total_time"])
# plot_time_per_epoch_comparison(results_adam["timestamps"], results_dg["timestamps"])