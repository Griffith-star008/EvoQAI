from setuptools import setup, find_packages

setup(
    name="AutoQuaHPC",
    version="1.0.0",
    author="Huy Ngo Anh",
    author_email="huyngoanh3@gmail.com",
    description="Self-Evolving Quantum Machine Learning Framework for Adaptive AIOT",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Griffith-star008/PhD_AutoQuaHPC",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.23.0",
        "scipy>=1.9.0",
        "matplotlib>=3.7.0"
    ],
)
