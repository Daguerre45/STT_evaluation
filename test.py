import os
import sys

if sys.platform == "win32":
    venv_base = os.path.dirname(os.path.dirname(sys.executable))
    cublas_dir = os.path.join(venv_base, "Lib", "site-packages", "nvidia", "cublas", "bin")
    cudnn_dir = os.path.join(venv_base, "Lib", "site-packages", "nvidia", "cudnn", "bin")
    
    if os.path.exists(cublas_dir):
        os.add_dll_directory(cublas_dir)
    if os.path.exists(cudnn_dir):
        os.add_dll_directory(cudnn_dir)

import torch

print(f"¿CUDA disponible?: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Dispositivo activo: {torch.cuda.get_device_name(0)}")