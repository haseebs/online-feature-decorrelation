#CC Build instructions
#module load python/3.7 and  module load scipy-stack and module load mariadb
#module load mariadb-connector-c

cmake_minimum_required(VERSION 3.14)
project(DecorrelationProject LANGUAGES CUDA CXX)
enable_testing()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

#Cedar
message( "Found environment variable SLURM_TMPDIR=$ENV{SLURM_TMPDIR}" )
set(CMAKE_CXX_FLAGS "-O3 -ltbb -lpthread")

add_executable(LinearCorrelation include/experiment/Experiment.h src/experiment/Experiment.cpp include/experiment/Database.h
        src/experiment/Database.cpp include/experiment/Metric.h src/experiment/Metric.cpp
        src/utils.cpp include/utils.h
        experiment_linear.cpp)

target_link_libraries(LinearCorrelation PUBLIC tbb)
target_link_libraries(LinearCorrelation PUBLIC mariadb)

Project(DecorrelationProject LANGUAGES CXX)

