from setuptools import setup, find_packages

setup(
    name="loop-algorithm-to-python-adaptive",  # distribution name (pip install ...)
    version="0.0.1",
    description="Adaptive wrapper around loop_to_python_api; currently backed by Sigridtt/LoopAlgorithmToPython.",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        #Can change to Miriam's repo later
        "loop_to_python_api @ git+https://github.com/Sigridtt/LoopAlgorithmToPython.git@main",
    ],
)