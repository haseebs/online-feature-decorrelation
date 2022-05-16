#include <iostream>
#include "include/utils.h"
#include "include/experiment/Experiment.h"
#include "include/experiment/Metric.h"
#include <string>
#include <random>
#include <cmath>
#include <algorithm>


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
  float alpha = my_experiment.get_float_param("alpha");
  float reg_lambda = my_experiment.get_float_param("regularizer_l");
  std::uniform_real_distribution<float> input_sampler(-10,10);
  std::uniform_real_distribution<float> weight_sampler(-2,2);
  std::vector<float> inputs;
  std::vector<float> weights;

  inputs.push_back(input_sampler(mt));
  inputs.push_back(inputs[0]);
  inputs.push_back(input_sampler(mt));

  weights.push_back(-199.5);
  weights.push_back(-200);
  weights.push_back(weight_sampler(mt));

  float target = inputs[2] * 5;
  float running_error = 0.05;

  for (int step = 0; step < my_experiment.get_int_param("steps"); step++) {

    inputs[0] = input_sampler(mt);
    inputs[1] = inputs[0];
    inputs[2] = input_sampler(mt);
    target = inputs[2] * 5;

    float pred = 0;
    for (int i = 0; i < inputs.size(); i++)
      pred += inputs[i] * weights[i];


    float error = pred - target;
    running_error = 0.99 * running_error + 0.01 * (target - pred);

    //std::vector<std::string> cur_error;
    //cur_error.push_back(std::to_string(my_experiment.get_int_param("run")));
    //cur_error.push_back(std::to_string(step));
    //cur_error.push_back(std::to_string(running_error));
    if (step%1 == 0){
      std::cout << "\nstep:" << step << std::endl;
      print_vector(inputs);
      print_vector(weights);
      std::cout << "target: " << target << " pred: " << pred;
      std::cout << std::endl;
    }
    for (int i = 0; i < inputs.size(); i++){
      //if (step % 1000 == 0)
      //  std::cout << "\n  change in:" << i << " : " << alpha * (error + reg_lambda*(weights[i]/weights[i])) * inputs[i] << std::endl;
      //weights[i] += alpha * error * inputs[i];
      //weights[i] += alpha * (error * inputs[i] - reg_lambda*(abs(weights[i])/weights[i]));
      weights[i] -= alpha * (error * inputs[i] + reg_lambda*(weights[i]));
    }
  }
}
