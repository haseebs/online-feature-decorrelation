#include <execution>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <set>
#include "../../../include/nn/networks/single_layer_network.h"
#include "../../../include/nn/neuron.h"
//
SingleLayerNetwork::SingleLayerNetwork(float step_size,
                   int seed,
                   int no_of_input_features,
                   int no_of_intermediate_features,
                   bool is_target_network) {
  this->step_size = step_size;
  this->std_cap = 0.001;
  this->mt.seed(seed);
  std::uniform_real_distribution<float> weight_sampler(-1, 1);
  std::uniform_real_distribution<float> prob_sampler(0, 1);
  std::uniform_int_distribution<int> index_sampler(0, no_of_input_features-1);

  for (int i = 0; i < no_of_input_features; i++) {
    LinearNeuron *n = new LinearNeuron(true, false);
    this->input_neurons.push_back(n);
  }

//  for (int i = 0; i < 3; i++) {
//    ReluNeuron *new_feature = new ReluNeuron(false, false);
//    this->intermediate_neurons.push_back(new_feature);
//      Synapse *new_synapse = new Synapse(this->input_neurons[i],
//                                         new_feature,
//                                         1,
//                                         step_size);
//      this->input_synapses.push_back(new_synapse);
//  }

  for (int i = 0; i < no_of_intermediate_features; i++) {
    ReluNeuron *new_feature = new ReluNeuron(false, false);
    this->intermediate_neurons.push_back(new_feature);

    std::set<int> connection_indices;
    int max_connections = index_sampler(mt) + 1;
    while (connection_indices.size() < max_connections){
      int new_index = index_sampler(mt);
      if (connection_indices.contains(new_index))
        continue;
      connection_indices.insert(new_index);
      Synapse *new_synapse = new Synapse(this->input_neurons[new_index],
                                         new_feature,
                                         weight_sampler(mt),
                                         step_size);
      this->input_synapses.push_back(new_synapse);
    }
  }

  for (int counter = 0; counter < this->intermediate_neurons.size(); counter++) {
    for (int inner_counter = 0; inner_counter < this->intermediate_neurons[counter]->incoming_synapses.size(); inner_counter++) {
      std::cout << this->intermediate_neurons[counter]->incoming_synapses[inner_counter]->input_neuron->id << "\tto\t"
                << this->intermediate_neurons[counter]->id << std::endl;
    }
  }

  predictions = 0;
  bias = 0;
  bias_gradients = 0;
  for (int j = 0; j < this->intermediate_neurons.size(); j++) {
    if (is_target_network)
      prediction_weights.push_back(weight_sampler(mt) * 10);
    else
      prediction_weights.push_back(0);
    prediction_weights_gradient.push_back(0);
    avg_feature_value.push_back(0);
    feature_mean.push_back(0);
    feature_std.push_back(1);
  }
}

float SingleLayerNetwork::forward(std::vector<float> inputs) {
//  Set input neurons value
//  if(this->time_step%100000 == 99999)
//    this->step_size *= 0.8;
  // inputs can be larger than input_neurons.size(). we simply ignore the rest
  for (int i = 0; i < input_neurons.size(); i++) {
    this->input_neurons[i]->value = inputs[i];
  }

  std::for_each(
      std::execution::par_unseq,
      this->intermediate_neurons.begin(),
      this->intermediate_neurons.end(),
      [&](ReluNeuron *n) {
        n->update_value();
    });

  std::for_each(
      std::execution::par_unseq,
      this->intermediate_neurons.begin(),
      this->intermediate_neurons.end(),
      [&](ReluNeuron *n) {
        n->fire();
    });

  for (int counter = 0; counter < intermediate_neurons.size(); counter++) {
    feature_mean[counter] = feature_mean[counter] * 0.99999 + 0.00001 * intermediate_neurons[counter]->value;
//    std::cout << "Feature mean = " << feature_mean[counter] << std::endl;
    if (std::isnan(feature_mean[counter])) {
      std::cout << "feature value = " << intermediate_neurons[counter]->value << std::endl;
      exit(1);
    }
    float temp = (feature_mean[counter] - intermediate_neurons[counter]->value);
    feature_std[counter] = feature_std[counter] * 0.99999 + 0.00001 * temp * temp;
    if (feature_std[counter] < this->std_cap)
      feature_std[counter] = this->std_cap;
  }

  predictions = 0;
  for (int i = 0; i < prediction_weights.size(); i++) {
    predictions += prediction_weights[i] * (this->intermediate_neurons[i]->value - feature_mean[i]) / sqrt(feature_std[i]);
  }
  predictions += bias;
  return predictions;
}

void SingleLayerNetwork::zero_grad() {
  //for (int counter = 0; counter < intermediate_neurons.size(); counter++) {
  //  intermediate_neurons[counter]->zero_grad();
  //}

  for (int index = 0; index < intermediate_neurons.size(); index++) {
    prediction_weights_gradient[index] = 0;
  }

  bias_gradients = 0;
}


std::vector<float> SingleLayerNetwork::get_prediction_weights() {
  std::vector<float> my_vec;
  my_vec.reserve(prediction_weights.size());
  for (int index = 0; index < prediction_weights.size(); index++) {
    my_vec.push_back(prediction_weights[index]);
  }
  return my_vec;
}

std::vector<float> SingleLayerNetwork::get_prediction_gradients() {
  std::vector<float> my_vec;
  my_vec.reserve(prediction_weights_gradient.size());
  for (int index = 0; index < prediction_weights_gradient.size(); index++) {
    my_vec.push_back(prediction_weights_gradient[index]);
  }
  return my_vec;
}

void SingleLayerNetwork::backward() {

  for (int index = 0; index < prediction_weights.size(); index++)
    prediction_weights_gradient[index] += (intermediate_neurons[index]->value - feature_mean[index]) / sqrt(feature_std[index]);

  bias_gradients += 1;
}

void SingleLayerNetwork::update_parameters(float error) {
  for (int index = 0; index < prediction_weights.size(); index++)
      prediction_weights[index] += prediction_weights_gradient[index] * error * step_size;

//  bias += error * step_size * 0.001 * bias_gradients;

}

float SingleLayerNetwork::read_output_values() {

  return predictions;
}

SingleLayerNetwork::~SingleLayerNetwork() {};
