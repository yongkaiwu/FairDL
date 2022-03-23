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
        'numpy>=1.17.1',
        'torch>=1.2.0',
        'scipy>=1.3.1',
        'torchvision>=0.4.0',
        'scikit_learn>=0.22.1'
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
