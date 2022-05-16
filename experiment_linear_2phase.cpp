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

/*
  Metric error_metric = Metric(my_experiment.database_name, "error_table",
                               std::vector < std::string > {"run", "step", "error", "td_error"},
                               std::vector < std::string > {"int", "int", "real", "real"},
                               std::vector < std::string > {"run", "step"});

  Metric avg_error = Metric(my_experiment.database_name, "predictions",
                            std::vector < std::string > {"run", "step", "x0", "x1", "x2", "x3", "x4", "x5", "x6", "pred", "target"},
                            std::vector < std::string > {"int", "int", "real" , "real", "real", "real", "real", "real", "real", "real", "real"},
                            std::vector < std::string > {"run", "step"});
  Metric network_state = Metric(my_experiment.database_name, "network_state",
                                std::vector < std::string > {"run", "step", "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9"},
                                std::vector <std::string > {"int", "int", "real", "real", "real", "real", "real", "real", "real", "real", "real", "real"},
                                std::vector <std::string> {"run", "step"});
*/
  std::cout << "Program started \n";

  std::mt19937 mt(my_experiment.get_int_param("seed"));
  float alpha = my_experiment.get_float_param("alpha");
  std::uniform_real_distribution<float> input_sampler(-10,10);
  std::uniform_real_distribution<float> weight_sampler(-2,2);
  std::vector<float> inputs;
  std::vector<float> weights;

  inputs.push_back(input_sampler(mt));
  inputs.push_back(inputs[0]);
  inputs.push_back(input_sampler(mt));

  weights.push_back(weight_sampler(mt));
  weights.push_back(weight_sampler(mt));
  weights.push_back(weight_sampler(mt));

  float target = inputs[0] * 5;

  float running_error = 0.05;
  for (int step = 0; step < my_experiment.get_int_param("steps"); step++) {

    inputs[0] = input_sampler(mt);
    inputs[1] = inputs[0];
    inputs[2] = input_sampler(mt);

    if(step  >= my_experiment.get_int_param("freq") - 1)
      target = inputs[2] * 5;
    else
      target = inputs[0] * 5;

    float pred = 0;
    for (int i = 0; i < inputs.size(); i++)
      pred += inputs[i] * weights[i];


    float error = target - pred;

    //l1 regularization
    float sum_of_weights = 0;
    float lambda = my_experiment.get_float_param("regularizer_l");
    for (int i = 0; i < weights.size(); i++)
      sum_of_weights += weights[i];
    //error += lambda * sum_of_weights;

    std::cout << "\nstep:" << step << std::endl;
    print_vector(inputs);
    print_vector(weights);
    std::cout << "target: " << target << " pred: " << pred;
    for (int i = 0; i < inputs.size(); i++){
      std::cout << "\n  change in:" << i << " : " << alpha * (error + lambda*(weights[i]/weights[i])) * inputs[i] << std::endl;
      weights[i] += alpha * (error + lambda*(weights[i]/weights[i])) * inputs[i];
      //weights[i] += alpha * (error + lambda*weights[i]) * inputs[i];
    }


    //float real_error = (real_target - pred)*(real_target - pred);
    //running_error = running_error*0.99999 + 0.00001*real_error;
  }
  //network_state.commit_values();
}