from setuptools import setup, find_packages

setup(
    name="python-custom-ops-bn",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "torch>=2.4.0",  # torch.library.custom_op
    ],
    python_requires=">=3.8",
    description="Custom BatchNorm operator implementation using PyTorch tensor operations",
)