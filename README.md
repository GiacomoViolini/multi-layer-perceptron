# Neural Network for MNIST-like Dataset - Python, C, CUDA

This project implements a **feedforward neural network** in **Python**, **C** and **CUDA** for image classification on the MNIST dataset. Each version builds upon the previous one, from a  CPU implementation to an optimized GPU training version using cuBLAS.

---

## Requirements

### Python Version

- Python 3.x
- Packages: `numpy`, `pandas`

### C Version

- GCC (or compatible C compiler)
- Standard libraries: `<stdio.h>`, `<stdlib.h>`, `<math.h>`, `<time.h>`

### CUDA Version

- NVIDIA CUDA Toolkit 11.0+
- cuBLAS library (included with CUDA Toolkit)
- NVIDIA GPU

---

## How to Run

### Python Version

**Compile:**

```bash
python -m venv venv
.\venv\Scripts\activate (Linux: source venv/bin/activate)
pip install --upgrade pip
pip install numpy pandas
python 1.mnist.py
```

### C Version

**Compile:**

```bash
gcc -Ofast -march=native -flto -funroll-loops -o 2.mnist 2.mnist.c -lm
```

**Run:**

```bash
./2.mnist
```

**Compile:**

```bash
gcc -Ofast -march=native -mtune=native -flto -funroll-loops -fno-plt -o 3.mnist-blas 3.mnist-blas.c -lopenblas -lm
```

**Run:**

```bash
./3.mnist-blas
```

**Compile:**

```bash
gcc -Ofast -fopenmp -march=native -mtune=native -flto -funroll-loops -fno-plt -o 4.mnist-blas-optimized 4.mnist-blas-optimized.c -lopenblas -lm
```

**Run:**

```bash
./4.mnist-blas-optimized
```

### CUDA Version

**Compile:**

```bash
nvcc -O3 -Xcompiler "-Ofast -march=native" --use_fast_math -arch=native -o 5.mnist 5.mnist.cu
```

**Run:**

```bash
./5.mnist
```

**Compile:**

```bash
nvcc -O3 -Xcompiler "-Ofast -march=native" --use_fast_math -arch=native -o 6.mnist-optimized 6.mnist-optimized.cu
```

**Run:**

```bash
./6.mnist-optimized
```

**Compile:**

```bash
 nvcc -O3 -Xcompiler "-Ofast -march=native" --use_fast_math -arch=native -o 7.mnist-optimized-cublas 7.mnist-optimized-cublas.cu -lcublas
```

**Run:**

```bash
./7.mnist-optimized-cublas
```


## Kaggle

The code can be accessed and run https://www.kaggle.com/code/giacomoviolini/multi-layer-perceptron 