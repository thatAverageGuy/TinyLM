# Ubuntu & CUDA Setup

## Hardware

```
GPU:    NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM, Ada Lovelace architecture)
OS:     Ubuntu 24.04.3 LTS (Noble Numbat)
Kernel: 6.17.0-14-generic
Driver: 590.48.01
```

## Verify your driver

If you're setting this up fresh, install the NVIDIA driver first. On Ubuntu 24.04:

```bash
sudo apt update
sudo ubuntu-drivers install
sudo reboot
```

After reboot, verify:

```bash
nvidia-smi
```

You should see something like this:

```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 590.48.01    Driver Version: 590.48.01    CUDA Version: 13.1               |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                Persistence-M  | Bus-Id          Disp.A | Volatile Uncorr. ECC |
|   0  NVIDIA GeForce RTX 4050 ...   Off  | 00000000:01:00.0   On  |                  N/A |
| N/A   38C    P8    2W / 60W             |    60MiB / 6141MiB     |     39%      Default |
+-----------------------------------------------------------------------------------------+
```

Three things to check:

- Driver version is listed — means NVIDIA driver is loaded correctly
- GPU name appears — means the GPU is detected
- Memory shows available VRAM — means the GPU is healthy

---

## The CUDA situation — read this carefully

This tripped us up and is worth understanding properly.

Running `nvidia-smi` shows **CUDA Version: 13.1** in the top right. This is **not** the CUDA toolkit version — it's the maximum CUDA version your driver *supports*. It's a capability ceiling, not what's actually installed.

When we ran:

```bash
ls /usr/local/ | grep cuda
# cuda
# cuda-13
# cuda-13.0
```

We found CUDA 13.0 toolkit installed. That's very new. PyTorch stable builds don't have wheels for CUDA 13.x yet.

**But here's the key insight: PyTorch doesn't use your system CUDA toolkit.**

PyTorch ships with its own CUDA runtime bundled inside the wheel. When you install PyTorch, it brings `libcudart`, `libcublas`, and everything else it needs. Your system CUDA toolkit is only relevant if you're compiling custom C++/CUDA extensions from source.

So the actual CUDA situation for this project is:

| What | Version | Where |
|------|---------|--------|
| System CUDA toolkit | 13.0 | `/usr/local/cuda-13.0` |
| PyTorch bundled CUDA runtime | 12.8 | Inside PyTorch wheel |
| CUDA your code actually runs on | 12.8 | PyTorch's bundled runtime |

Your driver (590.x) supports CUDA runtime 12.8 without any issues — driver support is forward-compatible within major versions.

---

## Verify CUDA is working with PyTorch

After installing PyTorch (covered in the next section), run:

```python
import torch

print(torch.__version__)         # 2.10.0+cu128
print(torch.cuda.is_available()) # True
print(torch.cuda.get_device_name(0)) # NVIDIA GeForce RTX 4050 Laptop GPU
print(torch.cuda.get_device_properties(0).total_memory // 1024**3) # 6 (GB)
```

All four lines should return meaningful values. If `torch.cuda.is_available()` returns `False`, the driver is either not installed or not visible to PyTorch.

---

## cuDNN

cuDNN 9.13.1 came pre-installed on this system alongside CUDA 13.0:

```bash
dpkg -l | grep cudnn
# cudnn9-jit-cuda-13  9.13.1.26-1
```

PyTorch also bundles its own cuDNN, so this is again not directly used by PyTorch — but it's good to have for future custom kernel work.

---

## What's next

With the driver verified and CUDA confirmed working through PyTorch, we set up Python and the package manager. → [Python & uv](python.md)