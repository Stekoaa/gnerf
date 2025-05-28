# Official code for GNeRF

## Cloning the Repository
The repository contains submodules, thus please check it out with
```bash
# clone the repo with submodules.
git clone --recursive https://github.com/theialab/lagrangian_hashes.git
```

## Installation

**Dependence**: Please install [Pytorch](https://pytorch.org/get-started/previous-versions/) first. This code is tested with Pytorch-2.1.0 with CUDA 12.1

Then please install nerfacc and lagrangian hashes code by running the setup.py file
``` bash
python -m pip install .
cd laghash
python -m pip install -e .
cd ..
```


Then to install other libraries(including tcnn), use the requirements.txt
``` bash
pip install -r requirements.txt
```

Then install [Faiss](https://github.com/facebookresearch/faiss), and [OptiX](https://developer.nvidia.com/designworks/optix/downloads/legacy) 7.6 required for KNN algorithms.

Next build CUDA code
``` bash
cd gnerf/lagrangian_hash/knn

# You may need tor replace paths in KNN.cu file

# Optix Compile
nvcc -ptx -arch=sm_86 -o shaders.cu.ptx shaders.cu -I/workspace/gnerf/NVIDIA-OptiX-SDK-7.6.0-linux64-x86_64/include -I/usr/local/cuda/include

#Cuda Compile
nvcc -Xcompiler -fPIC -c KNN.cu -o KNN.o \
  --gpu-architecture=compute_86 --gpu-code=sm_86 \
  -I/workspace/gnerf/NVIDIA-OptiX-SDK-7.6.0-linux64-x86_64/include \
  -I/usr/local/cuda/include \
  -D_GLIBCXX_USE_CXX11_ABI=0

# Bindings Coompile
g++ -shared -fPIC bindings.cpp KNN.o -o optix_knn$(python3-config --extension-suffix) \
  -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 \
  $(python3 -m pybind11 --includes) \
  -I/usr/local/cuda/include \
  -I/workspace/gnerf/NVIDIA-OptiX-SDK-7.6.0-linux64-x86_64/include \
  -I/usr/local/lib/python3.10/dist-packages/torch/include \
  -I/usr/local/lib/python3.10/dist-packages/torch/include/torch/csrc/api/include \
  -L/usr/local/lib/python3.10/dist-packages/torch/lib \
  -ltorch -ltorch_cpu -ltorch_python -lc10 -lcufft -lcuda \
  -Wl,-rpath=/usr/local/lib/python3.10/dist-packages/torch/lib

```

## Experiments 

Before running the example scripts, please check which dataset is needed, and download the dataset first. You could use `dataset.data_root` to specify the path or modify the config yaml.

### NeRF-Synthetic (blender) dataset

``` bash
python gnerf/train.py gnerf --dataset.scene lego
```

### Tanks and Temples (masked) dataset

``` bash
# Currently not implemented
python gnerf/train.py gnerf --dataset.scene Family
```

### For help with parameters

``` bash
python gnerf/train.py gnerf --help
```



## Citation

```bibtex
@inproceedings{govindarajan2024laghashes,
  title     = {Lagrangian Hashing for Compressed Neural Field Representations},
  author    = {Shrisudhan Govindarajan, Zeno Sambugaro, Ahan Shabhanov, Towaki Takikawa, Weiwei Sun, Daniel Rebain, Nicola Conci, Kwang Moo  Yi, Andrea Tagliasacchi},
  booktitle = {ECCV},
  year      = {2024},
}
```
