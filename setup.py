from setuptools import find_packages, setup

setup(
    name="nakta-model",
    version="0.0.2",
    packages=find_packages(exclude=["llama", "llama_org", "llama_vllm"]),
)
