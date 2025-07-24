import setuptools
import pathlib

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# Read the README for the long description
README = (HERE / "README.md").read_text(encoding="utf-8")

setuptools.setup(
    name="bbt-tuner",                                
    version="0.1.0",                                 
    author="Abdulvahap Mutlu",
    author_email="abdulvahapmutlu1@gmail.com",
    description="Adaptive Top-2 Bounding-Box Hyperparameter Tuner",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/abdulvahapmutlu/bounding-box-tuner-bbt",
    packages=setuptools.find_packages(where="."),    
    install_requires=[
        "numpy>=1.21",
        "scipy>=1.7"
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",                     
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",                               
)
