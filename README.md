# Neural Network for MNIST-like Dataset - Python, C, CUDA

This project implements a **feedforward neural network** in both **Python**, **C** and **CUDA** for image classification on the MNIST dataset.

---

## Requirements

### Python Version

- Python 3.x
- Packages: `numpy`, `pandas`

### C Version

- GCC (or compatible C compiler)
- Standard libraries: `<stdio.h>`, `<stdlib.h>`, `<math.h>`, `<time.h>`

---

## How to Run

### Python Version

**Compile:**

```bash
python -m venv venv
.\venv\Scripts\activate
pip install --upgrade pip
pip install numpy pandas
python 1.mnist.py
```

**Run:**

```bash
./2.mnist
```

### C Version

**Compile:**

```bash
gcc -o 2.mnist 2.mnist.c
```

**Run:**

```bash
./2.mnist
```

