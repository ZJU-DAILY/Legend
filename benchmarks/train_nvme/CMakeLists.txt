cmake_minimum_required (VERSION 3.1)
project (libnvm-benchmarks)

set (THREADS_PREFER_PTHREAD_FLAG ON)
find_package (Threads REQUIRED)

find_package (CUDA 8.0 REQUIRED)

make_sisci_benchmark (train-nvme-smartio train-nvme "main.cu;buffer.cu;ctrl.cc;settings.cu;queue.cu")

make_benchmark (train-nvme-module train-nvme "main.cu;buffer.cu;ctrl.cc;settings.cu;queue.cu")

make_benchmark_choice (train-nvme train-nvme-smartio train-nvme-module)
