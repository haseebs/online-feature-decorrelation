#CC Build instructions
#module load python/3.7 and  module load scipy-stack and module load mariadb
#module load mariadb-connector-c

cmake_minimum_required(VERSION 3.14)
project(OnlineFeatureDecorrelation LANGUAGES CUDA CXX)
enable_testing()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

#Cedar
message( "Found environment variable SLURM_TMPDIR=$ENV{SLURM_TMPDIR}" )
set(CMAKE_CXX_FLAGS "-O3 -ltbb -lpthread")

#add_executable(LinearCorrelation include/experiment/Experiment.h src/experiment/Experiment.cpp include/experiment/Database.h
#        src/experiment/Database.cpp include/experiment/Metric.h src/experiment/Metric.cpp
#        src/utils.cpp include/utils.h
#        experiment_linear.cpp)


add_executable(Count include/json.hpp include/utils.h src/utils.cpp include/experiment/Experiment.h include/experiment/Metric.h include/experiment/Database.h src/experiment/Metric.cpp src/experiment/Database.cpp src/experiment/Experiment.cpp
        count_configs.cpp)
target_link_libraries(Count PUBLIC tbb)
target_link_libraries(Count PUBLIC mariadb)

add_executable(SingleLayerGenerator include/experiment/Experiment.h src/experiment/Experiment.cpp include/experiment/Database.h
        src/experiment/Database.cpp include/experiment/Metric.h src/experiment/Metric.cpp
        src/utils.cpp include/utils.h src/nn/utils.cpp include/nn/utils.h
        include/nn/networks/neural_network.h include/nn/networks/single_layer_network.h
        src/nn/networks/neural_network.cpp src/nn/networks/single_layer_network.cpp
        include/nn/dynamic_elem.h include/nn/message.h include/nn/neuron.h include/nn/synapse.h
        src/nn/dynamic_elem.cpp src/nn/message.cpp src/nn/neuron.cpp src/nn/synapse.cpp
        experiment_single_layer_generator.cpp)

add_executable(DenseNetwork include/experiment/Experiment.h src/experiment/Experiment.cpp include/experiment/Database.h
        src/experiment/Database.cpp include/experiment/Metric.h src/experiment/Metric.cpp
        src/utils.cpp include/utils.h src/nn/utils.cpp include/nn/utils.h
        include/nn/networks/neural_network.h include/nn/networks/single_layer_network.h
        src/nn/networks/neural_network.cpp src/nn/networks/single_layer_network.cpp
        include/nn/networks/dense_network.h src/nn/networks/dense_network.cpp
        include/nn/dynamic_elem.h include/nn/message.h include/nn/neuron.h include/nn/synapse.h
        src/nn/dynamic_elem.cpp src/nn/message.cpp src/nn/neuron.cpp src/nn/synapse.cpp
        experiment_endtoend.cpp)

      #add_executable(LearnAll include/experiment/Experiment.h src/experiment/Experiment.cpp include/experiment/Database.h
      #        src/experiment/Database.cpp include/experiment/Metric.h src/experiment/Metric.cpp
      #        src/utils.cpp include/utils.h src/nn/utils.cpp include/nn/utils.h
      #        include/nn/networks/neural_network.h include/nn/networks/single_layer_network.h
      #        src/nn/networks/neural_network.cpp src/nn/networks/single_layer_network.cpp
      #        include/nn/dynamic_elem.h include/nn/message.h include/nn/neuron.h include/nn/synapse.h
      #        src/nn/dynamic_elem.cpp src/nn/message.cpp src/nn/neuron.cpp src/nn/synapse.cpp
      #        experiment_learn_all.cpp)

target_link_libraries(SingleLayerGenerator PUBLIC tbb)
target_link_libraries(SingleLayerGenerator PUBLIC mariadb)

target_link_libraries(DenseNetwork PUBLIC tbb)
target_link_libraries(DenseNetwork PUBLIC mariadb)

#target_link_libraries(LearnAll PUBLIC tbb)
#target_link_libraries(LearnAll PUBLIC mariadb)

Project(OnlineFeatureDecorrelation LANGUAGES CXX)
