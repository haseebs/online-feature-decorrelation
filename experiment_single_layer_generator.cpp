#include <iostream>
#include "include/utils.h"
#include "include/experiment/Experiment.h"
#include "include/experiment/Metric.h"
#include <string>
#include <random>
#include <cmath>
#include <algorithm>
#include "include/nn/networks/single_layer_network.h"
#include "include/nn/neuron.h"


int main(int argc, char *argv[]) {

  Experiment my_experiment = Experiment(argc, argv);

  Metric error_metric = Metric(my_experiment.database_name, "error_table",
                               std::vector < std::string > {"run", "step", "error"},
                               std::vector < std::string > {"int", "int", "real"},
                               std::vector < std::string > {"run", "step"});

  Metric feature_metric = Metric(my_experiment.database_name, "feature_table",
                               std::vector < std::string > {"run", "step", "w1", "w2", "w3"},
                               std::vector < std::string > {"int", "int", "real", "real", "real"},
                               std::vector < std::string > {"run", "step"});

  std::cout << "Program started \n";


  std::mt19937 mt(my_experiment.get_int_param("seed"));
  int total_inputs = my_experiment.get_int_param("n_inputs") + my_experiment.get_int_param("n_distractors");
  auto target_network = SingleLayerNetwork(0.0,
                                           my_experiment.get_int_param("seed") + 1000,
                                           my_experiment.get_int_param("n_inputs"),
                                           my_experiment.get_int_param("n_target_features"),
                                           true);
  Neuron::neuron_id_generator = 0;
  auto learning_network = SingleLayerNetwork(my_experiment.get_float_param("step_size"),
                                             my_experiment.get_int_param("seed"),
                                             total_inputs,
                                             my_experiment.get_int_param("n_learner_features"),
                                             false);

  NetworkVisualizer target_vis = NetworkVisualizer(&target_network);
  NetworkVisualizer learning_vis = NetworkVisualizer(&learning_network);
  target_vis.generate_dot(0);
  learning_vis.generate_dot(1);

  auto input_sampler = uniform_random(my_experiment.get_int_param("seed"), -10, 10);

  float running_error = 0.05;

  for (int step = 0; step < my_experiment.get_int_param("steps"); step++) {
    if (step % my_experiment.get_int_param("replace_every") == 1)
      learning_network.replace_features(my_experiment.get_float_param("replace_perc"));
    auto input = input_sampler.get_random_vector(total_inputs);
    float pred = learning_network.forward(input);
    float target = target_network.forward(input);
    float error = target - pred;

    running_error = 0.995 * running_error + 0.005 * (target - pred) * (target - pred);

    //std::vector<std::string> cur_error;
    //cur_error.push_back(std::to_string(my_experiment.get_int_param("run")));
    //cur_error.push_back(std::to_string(step));
    //cur_error.push_back(std::to_string(running_error));
    learning_network.backward();
    learning_network.update_parameters(error);
    if (step%1 == 0){
      std::cout << "\nstep:" << step << std::endl;
      print_vector(input);
      print_vector(learning_network.get_prediction_weights());
      print_vector(learning_network.get_feature_utilities());
      print_vector(learning_network.get_prediction_gradients());
      std::cout << "target: " << target << " pred: " << pred << std::endl;
      std::cout << "running err: " << running_error << std::endl;
    }
    learning_network.zero_grad();
  }
  learning_vis.generate_dot(my_experiment.get_int_param("steps"));
}
