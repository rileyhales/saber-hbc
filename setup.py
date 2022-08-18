from setuptools import setup

with open("README.md", "r") as readme:
    long_description = readme.read()

with open('requirements.txt', 'r') as req:
    install_requires = req.read().splitlines()

description = 'tools for hydrological bias correction on large models'
version = '0.4.0'

setup(
    name='saber',
    packages=['saber'],
    version=version,
    description=description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Riley Hales',
    license='BSD 3-Clause',
    license_family='BSD',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Hydrology',
        'Topic :: Scientific/Engineering :: Visualization',
        'Topic :: Scientific/Engineering :: GIS',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
    ],
    python_requires=">=3",
    install_requires=install_requires
)
