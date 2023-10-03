FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

# Set a working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN cd app
RUN pip install .
RUN cd ..

# Uninstall triton and install triton-nightly
RUN pip uninstall -y triton && \
    pip install -U --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly

# Clone the lm-evaluation-harness repository and install it
RUN git clone https://github.com/EleutherAI/lm-evaluation-harness && \
    cd lm-evaluation-harness && \
    pip install -e .
