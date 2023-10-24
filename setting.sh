cd ..
# set cuda toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda-repo-ubuntu2204-12-1-local_12.1.0-530.30.02-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-1-local_12.1.0-530.30.02-1_amd64.deb
cp /var/cuda-repo-ubuntu2204-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/
apt-get update
apt-get -y install cuda

# install python packages
pip install fairscale
pip install sentencepiece
pip install datasets
pip install matplotlib
pip install fire
pip uninstall -y triton 
pip install -U --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly==2.1.0.dev20230822000928

# install custom kernel
git clone https://github.com/AI-CE-2023/flash.git
cd flash
make install
cd ..

# install nakta
cd nakta
pip install .

# install programs
apt-get update
apt-get -y install bc
apt-get update
apt-get -y install tmux