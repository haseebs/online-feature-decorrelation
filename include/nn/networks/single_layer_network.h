#ifndef INCLUDE_NN_NETWORKS_SINGLE_LAYER_NETWORK_H_
#define INCLUDE_NN_NETWORKS_SINGLE_LAYER_NETWORK_H_


#include <vector>
#include <map>
#include <random>
#include <string>
#include <map>
#include "../synapse.h"
#include "../neuron.h"
#include "../dynamic_elem.h"
#include "./neural_network.h"

class SingleLayerNetwork {
 protected:
  int64_t time_step;
  std::mt19937 mt;

 public:
//
//

  float step_size;
  float predictions;
  float bias;
  float bias_gradients;
  float std_cap;

  std::vector<Synapse*> input_synapses;

  std::vector<float> prediction_weights;

  std::vector<float> feature_mean;

  std::vector<float> feature_std;

  std::vector<float> feature_utility_trace;

  std::vector<float> prediction_weights_gradient;

  float  get_target_without_sideeffects(std::vector<float> inputs);

  std::vector<Neuron*> input_neurons;

  std::vector<SigmoidNeuron*> intermediate_neurons;

  float read_output_values();

  void replace_features(float perc_to_replace);
  SingleLayerNetwork(float step_size, int seed, int no_of_input_features, int no_of_intermediate_features, bool is_target_network);

  ~SingleLayerNetwork();

  float forward(std::vector<float> inputs);

  void zero_grad();

  void backward();

  void update_parameters(float error);

  std::vector<float> get_prediction_gradients();
  std::vector<float> get_prediction_weights();
  std::vector<float> get_feature_utilities();

  std::map<std::pair<int, int>, float> feature_correlations;
  void calculate_all_correlations();
  void print_all_correlations();
  void print_all_statistics();
  int count_highly_correlated_features();
  void replace_features_with_idx(int feature_idx);
  void decorrelate_features_baseline(int sum_features);

};



#endif //INCLUDE_NN_NETWORKS_SINGLE_LAYER_NETWORK_H_
