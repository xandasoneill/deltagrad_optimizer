from visualizations import load_and_plot_results, plot_accuracy_comparison, plot_variance_comparison, plot_learning_curves, plot_accuracy_evolution, plot_time_per_epoch_comparison
import joblib

results_adam=joblib.load("results/results_lrtest/results_values/3x_lr/Adam_results_batch16_lr0.0005221157825505848.pkl")
results_dg =joblib.load("results/results_lrtest/results_values/3x_lr/DeltaGrad_results_batch16_lr0.0958208553169726.pkl")
# plot_variance_comparison(results_dg,results_adam)
print(results_adam["total_time"])
print(results_dg["total_time"])
plot_time_per_epoch_comparison(results_adam["timestamps"], results_dg["timestamps"])