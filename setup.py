from setuptools import setup, find_packages

setup(
    name="nakta-model",
    version="0.0.1",
    packages=find_packages(exclude=["llama", "llama_org", "llama_vllm"]),
)
