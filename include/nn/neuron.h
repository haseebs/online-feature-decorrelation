//
// Created by Khurram Javed on 2021-09-20.
//

#ifndef INCLUDE_NN_SYNCED_NEURON_H_
#define INCLUDE_NN_SYNCED_NEURON_H_


#include <vector>
#include <queue>
#include <unordered_set>
#include <unordered_map>
#include <utility>
#include "./dynamic_elem.h"
#include "./synapse.h"
#include "./message.h"
#include "./utils.h"

class Neuron : public dynamic_elem {
 public:
  bool frozen;
  bool is_recurrent_neuron;
  static int64_t neuron_id_generator;
  static std::mt19937 gen;
  bool is_input_neuron;
  bool is_bias_unit;
  int layer_number;
  float value;
  float pre_sync_value;
  int drinking_age;
  float value_before_firing;
  float neuron_utility;
  float neuron_utility_to_distribute;
  float sum_of_utility_traces;
  float running_mean;
  float running_variance;
  bool is_output_neuron;
  bool useless_neuron;
  int64_t id;
  float neuron_utility_trace_decay_rate;
  int neuron_age;

  void set_layer_number(int layer);

  int get_layer_number();

  void forward_gradients();

  virtual void update_value();



  message error_gradient;
  std::vector<Synapse *> outgoing_synapses;
  std::vector<Synapse *> incoming_synapses;

  Neuron(bool is_input, bool is_output);

  virtual void fire();

  float introduce_targets(float target);

  virtual float backward(float output_grad) = 0;

  virtual float forward(float temp_value) = 0;

  void update_utility();

  float get_utility();

  bool is_mature();

  ~Neuron() = default;
};

class RecurrentRelu : public Neuron {

 public:
  float old_value;

  bool learning = true;

  void disable_learning();

  void enable_learning();

  void compute_gradient_of_all_synapses(std::vector<float> prediction_error_list);

  void update_value();

  float backward(float output_grad);

  float forward(float temp_value);

  Synapse* recurrent_synapse;

  RecurrentRelu(bool is_input, bool is_output);

  void fire();


};

class LSTM : public Neuron{

//  Variables for making predictions
  std::vector<float> w_i;
  std::vector<float> w_f;
  std::vector<float> w_g;
  std::vector<float> w_o;
  std::vector<float> input_means;
  std::vector<float> input_std;


  float std_cap;

  float u_i, u_f, u_g, u_o;
  float b_i, b_f, b_g, b_o;
  float c;
  float h;
  float old_c;
  float old_h;
  float i_val;
  float f;
  float g;
  float o;

//  Variables for computing the gradient
  std::vector<float> Hw_i;
  std::vector<float> Hw_f;
  std::vector<float> Hw_g;
  std::vector<float> Hw_o;

  std::vector<float> Cw_i;
  std::vector<float> Cw_f;
  std::vector<float> Cw_g;
  std::vector<float> Cw_o;

  float Hu_i, Hu_f, Hu_g, Hu_o;
  float Cu_i, Cu_f, Cu_g, Cu_o;

  float Hb_i, Hb_f, Hb_g, Hb_o;
  float Cb_i, Cb_f, Cb_g, Cb_o;


//  Variable for storing gradients
  std::vector<float> Gw_i;
  std::vector<float> Gw_f;
  std::vector<float> Gw_g;
  std::vector<float> Gw_o;

  float Gu_i, Gu_f, Gu_g, Gu_o;

  float Gb_i, Gb_f, Gb_g, Gb_o;

  int users;

  float copy_of_h;


 public:

  void update_statistics();

  std::vector<float> get_normalized_values();

  std::vector<Neuron *> incoming_neurons;

  float get_value_without_sideeffects();

  int get_users();

  void decay_gradient(float decay_rate);

  void increment_user();

  void decrement_user();

  void update_value_delay();

  void update_value_sync();

  void reset_state();

  void zero_grad();

  void accumulate_gradient(float incoming_grad);

  void print_gradients();

  float get_hidden_state();

  void update_weights(float step_size);

  void update_weights(float step_size, float error);

  void add_synapse(Neuron* s, float w_i, float w_f, float w_g, float w_o);

  float old_value;

  bool learning = true;

  void disable_learning();

  void enable_learning();

  void compute_gradient_of_all_synapses();

  void update_value();

  float backward(float output_grad);

  float forward(float temp_value);

  Synapse* recurrent_synapse;

  LSTM(float ui, float uf, float ug, float uo, float bi, float bf, float bg, float bo, float std_cap);

  void fire();


};

class BiasNeuron : public Neuron {
 public:
  BiasNeuron();
  float backward(float output_grad);

  float forward(float temp_value);
};


class LinearNeuron : public Neuron {
 public:
  float backward(float output_grad);

  float forward(float temp_value);

  LinearNeuron(bool is_input, bool is_output);
};

class ReluNeuron: public Neuron {
 public:
  float backward(float output_grad);

  float forward(float temp_value);

  ReluNeuron(bool is_input, bool is_output);
};

class SigmoidNeuron : public Neuron {
 public:
  float backward(float output_grad);

  float forward(float temp_value);

  SigmoidNeuron(bool is_input, bool is_output);
};



#endif //INCLUDE_NN_SYNCED_NEURON_H_
