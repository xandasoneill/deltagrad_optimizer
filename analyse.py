# Copyright 2026 Alexandre de Abreu O'Neill Mendes

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from visualizations import load_and_plot_results, plot_accuracy_comparison, plot_variance_comparison, plot_learning_curves, plot_accuracy_evolution, plot_mean_time_per_epoch
import joblib

results_adam=joblib.load("results/results_datanoise/20%_50epochs/bs_512/results_values/Adam_results_batch512_lr0.00031705531640654854.pkl")
results_dg =joblib.load("results/results_datanoise/20%_50epochs/bs_512/results_values/DeltaGrad_results_batch512_lr0.27887247907205764.pkl")
# load_and_plot_results(results_dg, results_adam)
# print(results_adam["all_total_times"])
# print(results_dg["all_total_times"])
plot_mean_time_per_epoch(results_adam["all_timestamps"], results_dg["all_timestamps"])

