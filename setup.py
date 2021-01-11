import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="livapordata",
    version="0.9.0",
    author="Jacob Schwartz",
    author_email="jacobas@princeton.edu",
    description="Literature data on lithium, especially its vapor",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cfe316/livapordata",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: BSD License",
        "Intended Audience :: Science/Research",
    ],
    python_requires='>=3.6',
    install_requires=[
        "numpy >= 1.15",
        "scipy >= 1.2",
    ],
)
