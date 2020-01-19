# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import setuptools, platform

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="torch_testbed",
    version="0.2",
    author="Shital Shah",
    author_email="shitals@microsoft.com",
    description="Simple clean code to experiment with PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sytelus/torch_testbed",
    packages=setuptools.find_packages(),
	license='MIT',
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
    include_package_data=True,
    install_requires=['runstats']
)

