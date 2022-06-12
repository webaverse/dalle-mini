# AWS command
source activate pytorch_p39

#

pip install --upgrade pip

# Installs the wheel compatible with Cuda >= 11.4 and cudnn >= 8.2
pip install "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_releases.html

#

pip install -q dalle-mini
pip install -q git+https://github.com/patil-suraj/vqgan-jax.git
