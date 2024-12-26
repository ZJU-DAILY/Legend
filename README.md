Legend: A Lightweight Heterogeneous System for Out-of-core Graph Embedding Learning
===============================================================================
Legend is a lightweight heterogeneous system for efficient and cost-effective graph embedding learning, comprising a CPU, a GPU and an NVMe SSD. It adopts a novel workflow that reconsiders data placement and meticulously assigning tasks to leverage the unique strengths of each hardware component. A prefetch-friendly order is proposed to support embedding prefetching from NVMe SSD to GPU. Furthermore, it also optimize GPU-NVMe SSD direct access and GPU computing to achieve better performance. 

Environment preparation
-------------------------------------------------------------------------------
### Requiements ###
* Python 3.8
* CUDA 11.1
* torch 1.7.1
* Samsung 980 NVMe SSD
* Nvidia A100 GPU

### Compiling Nvidia Driver ###
The Nvidia driver kernel sources are typically installed in `/usr/src/`. Following commands are used to get the kernel symbols. 

```
$ cd /usr/src/nvidia-550.54.15
$ sudo make
```
### Unbinding NVMe Driver ###
The default NVMe driver should be unbind first before install the customized NVMe driver. The PCI ID of the NVMe SSD is required to do this, which can be find by using `lspci`. 
We assume the PCI ID is `86:00.0`. The NVMe driver can be unbinded using following commands. 

```
$ echo -n "0000:86:00.0" > /sys/bus/pci/devices/0000\:86\:00.0/driver/unbind
```

Project Building
-------------------------------------------------------------------------------
From the project root directory, do the following:

```
$ mkdir build; cd build
$ cmake
$ make libnvm
$ make benchmarks
```

After this, the `libnvm` kernel module have to be compiled. In the `build` directory, do the following:

```
$ cd module
$ make
```

Subsequently, we need to load the custom `libnvm` kernel module in the `module` directory. It can be loaded and unloaded with the following:

```
$ sudo make load
$ sudo make unload
```

This should create a `/dev/libnvm0` device file, representing the disk's BAR0. 

Datasets
-------------------------------------------------------------------------------
Each dataset can be obtained from the following links.

| Dataset | Nodes | Edges | Relations | Link                                          |
| ------- | ----------- | -------------- | -------------------- | --------------------------------------------- |
| FB15k   | 15k     | 592k           | 1345        | https://dl.fbaipublicfiles.com/starspace/fb15k.tgz  |
| LiveJournal   | 4.8M  | 68M              | -           | https://snap.stanford.edu/data/soc-LiveJournal1.txt.gz |
| Twitter  | 41.6M     | 1.46B            | - | https://snap.stanford.edu/data/twitter-2010.txt.gz    |
| Freebase86m     | 86.1M   | 304.7M            | 14824        | https://data.dgl.ai/dataset/Freebase.zip   |

Example Running
--------------------------------------------------------------------------------
The executable files are located in `build/bin`

To train without NVMe SSD, we can run the code with the followings commands. 

```
$ ./nvm-train-nonvme
```

The dataset used to train should be modified in the code `./benchmarks/train_nonvme/main.cu`.

To train with NVMe SSD, we can run the code with the followings commands. 

```
$ ./nvm-train-nvme --ctrl=/dev/libnvm0 --threads=4096 --page_size=32768 --queue_pairs=8 --queue_depth=1024
```

* `ctrl` is the path of the custom NVMe controller file. 
* `threads` specifies the threads used to load data from NVMe SSD. 
* `page_size` denotes the size of each IO request. 
* `queue_pairs` indicates the number of queues in the NVMe controller. 
* `queue_depth` specifies the queue depth per queue. 

The dataset used to train should be modified in the code `./benchmarks/train_nvme/main.cu`.

Acknowledgments
--------------------------------------------------------------------------------
The GPU-SSD direct access module of this project is built on top of an open-source codebase available [here](https://github.com/enfiskutensykkel/ssd-gpu-dma). We employ the framework of this codebase and develop a customized queue management mechanism (submission queue inserting, doorbell ringing, and completion queue polling) to improve the throughput between GPU and SSD. 
