#include <execution>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <set>
#include <utility>
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
//    SigmoidNeuron *new_feature = new SigmoidNeuron(false, false);
//    this->intermediate_neurons.push_back(new_feature);
//      Synapse *new_synapse = new Synapse(this->input_neurons[i],
//                                         new_feature,
//                                         1,
//                                         step_size);
//      this->input_synapses.push_back(new_synapse);
//  }

  for (int i = 0; i < no_of_intermediate_features; i++) {
    SigmoidNeuron *new_feature = new SigmoidNeuron(false, false);
    this->intermediate_neurons.push_back(new_feature);

    std::set<int> connection_indices;
    int max_connections = index_sampler(mt) + 1;
    while (connection_indices.size() < max_connections){
      int new_index = index_sampler(mt);
      if (connection_indices.contains(new_index))
        continue;
      connection_indices.insert(new_index);
      float new_weight = weight_sampler(mt);
      Synapse *new_synapse = new Synapse(this->input_neurons[new_index],
                                         new_feature,
                                         new_weight,
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
    feature_utility_trace.push_back(0); // trace of normalized feature value * outgoing weight
    feature_mean.push_back(0);
    feature_std.push_back(1);
  }
}

void SingleLayerNetwork::decorrelate_features_baseline() {
  for (int i = 0; i < intermediate_neurons.size(); i++) {
    bool replaced = false;
    for (int j = i+1; j < intermediate_neurons.size(); j++) {
      if (replaced)
        continue;
      auto id_pair = std::make_pair(intermediate_neurons[i]->id, intermediate_neurons[j]->id);
      if (fabs(feature_correlations[id_pair]) > 0.85 && intermediate_neurons[i]->neuron_age > 190000 && intermediate_neurons[j]->neuron_age > 190000 ){
        prediction_weights[j] += prediction_weights[i]; //TODO assume single outgoing w
        replace_features_with_idx(i);
        replaced = true;
      }
    }
  }
}


void SingleLayerNetwork::replace_features_with_idx(int feature_idx) {
  std::uniform_real_distribution<float> weight_sampler(-1, 1);
  std::uniform_real_distribution<float> prob_sampler(0, 1);
  std::uniform_int_distribution<int> index_sampler(0, this->input_neurons.size() - 1);
  float median_feature_utility = median(feature_utility_trace);

  for ( auto & synapse: this->intermediate_neurons[feature_idx]->incoming_synapses )
    synapse->is_useless = true;

  SigmoidNeuron *new_feature = new SigmoidNeuron(false, false);
  this->intermediate_neurons[feature_idx] = new_feature;

  std::set<int> connection_indices;
  int max_connections = index_sampler(mt) + 1;
  while (connection_indices.size() < max_connections){
    int new_index = index_sampler(mt);
    if (connection_indices.contains(new_index))
      continue;
    connection_indices.insert(new_index);
    float new_weight = weight_sampler(mt);
    Synapse *new_synapse = new Synapse(this->input_neurons[new_index],
                                       new_feature,
                                       new_weight,
                                       step_size);
    this->input_synapses.push_back(new_synapse);
  }
  prediction_weights[feature_idx] = 0;
  prediction_weights_gradient[feature_idx] = 0;
  feature_utility_trace[feature_idx] = median_feature_utility;
  feature_mean[feature_idx] = 0;
  feature_std[feature_idx] = 1;

  for (int i = 0; i < intermediate_neurons.size(); i++) {
    auto id_pair = std::make_pair(intermediate_neurons[i]->id, intermediate_neurons[feature_idx]->id);
    feature_correlations.erase(id_pair);
    id_pair = std::make_pair(intermediate_neurons[feature_idx]->id, intermediate_neurons[i]->id);
    feature_correlations.erase(id_pair);
  }

  auto it = std::remove_if(this->input_synapses.begin(), this->input_synapses.end(), [](Synapse *s){return s->is_useless;});
  this->input_synapses.erase(it, this->input_synapses.end());

  std::for_each(
      std::execution::seq,
      this->input_neurons.begin(),
      this->input_neurons.end(),
      [&](Neuron *n) {
        auto it = std::remove_if(n->outgoing_synapses.begin(), n->outgoing_synapses.end(), [](Synapse *s){return s->is_useless;});
        n->outgoing_synapses.erase(it, n->outgoing_synapses.end());
    });
}


void SingleLayerNetwork::replace_features(float perc_to_replace) {
  std::uniform_real_distribution<float> weight_sampler(-1, 1);
  std::uniform_real_distribution<float> prob_sampler(0, 1);
  std::uniform_int_distribution<int> index_sampler(0, this->input_neurons.size() - 1);
  float median_feature_utility = median(feature_utility_trace);
  //std::vector<Neuron*> replaced_features;

  for (int i = 0; i < int(prediction_weights.size() * perc_to_replace); i++) {
    float least_useful_idx = min_idx(feature_utility_trace);
    //replaced_features.push_back(intermediate_neurons[least_useful_feature_idx]); //TODO cleanup
    for ( auto & synapse: this->intermediate_neurons[least_useful_idx]->incoming_synapses )
      synapse->is_useless = true;

    SigmoidNeuron *new_feature = new SigmoidNeuron(false, false);
    this->intermediate_neurons[least_useful_idx] = new_feature;

    std::set<int> connection_indices;
    int max_connections = index_sampler(mt) + 1;
    while (connection_indices.size() < max_connections){
      int new_index = index_sampler(mt);
      if (connection_indices.contains(new_index))
        continue;
      connection_indices.insert(new_index);
      float new_weight = weight_sampler(mt);
      Synapse *new_synapse = new Synapse(this->input_neurons[new_index],
                                         new_feature,
                                         new_weight,
                                         step_size);
      this->input_synapses.push_back(new_synapse);
    }
    prediction_weights[least_useful_idx] = 0;
    prediction_weights_gradient[least_useful_idx] = 0;
    feature_utility_trace[least_useful_idx] = median_feature_utility;
    feature_mean[least_useful_idx] = 0;
    feature_std[least_useful_idx] = 1;
    for (int i = 0; i < intermediate_neurons.size(); i++) {
      auto id_pair = std::make_pair(intermediate_neurons[i]->id, intermediate_neurons[least_useful_idx]->id);
      feature_correlations.erase(id_pair);
      id_pair = std::make_pair(intermediate_neurons[least_useful_idx]->id, intermediate_neurons[i]->id);
      feature_correlations.erase(id_pair);
    }
  }

  //cleanup
  //for ( auto & feature : replaced_features) {
  //  for ( auto & synapse: feature->incoming_synapses )
  //    synapse->is_useless = true;

  auto it = std::remove_if(this->input_synapses.begin(), this->input_synapses.end(), [](Synapse *s){return s->is_useless;});
  this->input_synapses.erase(it, this->input_synapses.end());

  std::for_each(
      std::execution::seq,
      this->input_neurons.begin(),
      this->input_neurons.end(),
      [&](Neuron *n) {
        auto it = std::remove_if(n->outgoing_synapses.begin(), n->outgoing_synapses.end(), [](Synapse *s){return s->is_useless;});
        n->outgoing_synapses.erase(it, n->outgoing_synapses.end());
    });
}



float SingleLayerNetwork::forward(std::vector<float> inputs) {
  //TODO remove initial non-stationarity due to normalization in target network?
//  Set input neurons value
//  if(this->time_step%100000 == 99999)
//    this->step_size *= 0.8;
  // inputs can be larger than input_neurons.size(). we simply ignore the rest
  for (int i = 0; i < input_neurons.size(); i++) {
    this->input_neurons[i]->value = inputs[i];
  }

  std::for_each(
      std::execution::seq,
      this->intermediate_neurons.begin(),
      this->intermediate_neurons.end(),
      [&](SigmoidNeuron *n) {
        n->update_value();
    });

  std::for_each(
      std::execution::seq,
      this->intermediate_neurons.begin(),
      this->intermediate_neurons.end(),
      [&](SigmoidNeuron *n) {
        n->fire();
    });

  for (int counter = 0; counter < intermediate_neurons.size(); counter++) {
    feature_mean[counter] = feature_mean[counter] * 0.999 + 0.001 * intermediate_neurons[counter]->value;
    if (std::isnan(feature_mean[counter])) {
      std::cout << "feature value = " << intermediate_neurons[counter]->value << std::endl;
      exit(1);
    }
    float temp = (feature_mean[counter] - intermediate_neurons[counter]->value);
    feature_std[counter] = feature_std[counter] * 0.999 + 0.001 * temp * temp;
    if (feature_std[counter] < this->std_cap)
      feature_std[counter] = this->std_cap;
  }

  predictions = 0;
  for (int i = 0; i < prediction_weights.size(); i++) {
    float feature_output = prediction_weights[i] * (this->intermediate_neurons[i]->value - feature_mean[i]) / sqrt(feature_std[i]);
    feature_utility_trace[i] = feature_utility_trace[i] * 0.999 + 0.001 * fabs(feature_output);
    predictions += feature_output;
  }
  predictions += bias;
  return predictions;
}


void SingleLayerNetwork::calculate_all_correlations() {
  //TODO the correlations map needs to be cleaned up for removed features
  for (int i = 0; i < intermediate_neurons.size(); i++) {
    for (int j = i; j < intermediate_neurons.size(); j++) {
      auto id_pair = std::make_pair(intermediate_neurons[i]->id, intermediate_neurons[j]->id);
      float i_normalizaed = (intermediate_neurons[i]->value - feature_mean[i]) / sqrt(feature_std[i]);
      float j_normalizaed = (intermediate_neurons[j]->value - feature_mean[j]) / sqrt(feature_std[j]);
      if (feature_correlations.count(id_pair))
        feature_correlations[id_pair] = 0.95 * feature_correlations[id_pair] + 0.05 * i_normalizaed * j_normalizaed;
      else
        feature_correlations[id_pair] = 0.05 * i_normalizaed * j_normalizaed;
    }
  }
}

void SingleLayerNetwork::print_all_correlations() {
  std::cout << "feature correlations:" << std::endl;
  for (int i = 0; i < intermediate_neurons.size(); i++) {
    for (int j = i+1; j < intermediate_neurons.size(); j++) {
      auto id_pair = std::make_pair(intermediate_neurons[i]->id, intermediate_neurons[j]->id);
      if (fabs(feature_correlations[id_pair]) > 0.85 && intermediate_neurons[i]->neuron_age > 200000 && intermediate_neurons[j]->neuron_age > 200000 ){
        std::cout << "--->\t";
        std::cout << intermediate_neurons[i]->id << "\t->\t" << intermediate_neurons[j]->id << "\t:\t" << feature_correlations[id_pair] << std::endl;
      }
    }
  }
}

int SingleLayerNetwork::count_highly_correlated_features() {
  std::set<int> correlated_feature_ids;
  for (int i = 0; i < intermediate_neurons.size(); i++) {
    for (int j = i+1; j < intermediate_neurons.size(); j++) {
      auto id_pair = std::make_pair(intermediate_neurons[i]->id, intermediate_neurons[j]->id);
      if (fabs(feature_correlations[id_pair]) > 0.85 && intermediate_neurons[i]->neuron_age > 200000 && intermediate_neurons[j]->neuron_age > 200000 ){
        correlated_feature_ids.insert(intermediate_neurons[i]->id);
        correlated_feature_ids.insert(intermediate_neurons[j]->id);
      }
    }
  }
  return correlated_feature_ids.size();
}

void SingleLayerNetwork::print_all_statistics() {
  for (int i = 0; i < intermediate_neurons.size(); i++) {
    std::cout << intermediate_neurons[i]->id << "\tmean: " << feature_mean[i];
    std::cout << "\tstd: " << feature_std[i] << std::endl;
  }
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

std::vector<float> SingleLayerNetwork::get_feature_utilities() {
  std::vector<float> my_vec;
  my_vec.reserve(feature_utility_trace.size());
  for (int index = 0; index < feature_utility_trace.size(); index++) {
    my_vec.push_back(feature_utility_trace[index]);
  }
  return my_vec;
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
