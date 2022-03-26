from setuptools import find_packages, setup

with open("README.md", 'r') as fh:
    long_description = fh.read()

setup(
    name="fairdl",
    version="0.1",
    author='',
    maintainer='',
    description="A PyTorch library for fair machine learning and deep learning.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    url='',
    include_package_data=True,
    install_requires=[
        'torch>=1.10.0'
    ],
    keywords=['python', 'fairness', 'deep learning', 'pytorch', 'ai', 'ethics'],
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Libraries",
        "License :: OSI Approved :: MIT License"
    ],
    license="MIT",

)
