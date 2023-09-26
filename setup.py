from setuptools import find_packages, setup

setup(
    name="nakta-models",
    version="0.0.3",
    packages=find_packages(exclude=["llama", "llama_vllm"]),
)
