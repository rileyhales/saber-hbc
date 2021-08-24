from setuptools import setup

with open("README.md", "r") as readme:
    long_description = readme.read()

with open('requirements.txt', 'r') as req:
    install_requires = req.read().splitlines()

description = 'tools for bias correcting large scale hydrologic models limited by observed data'

setup(
    name='rbc',
    packages=['rbc'],
    version='0.1.0',
    description=description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Riley Hales',
    license='BSD 3-Clause',
    license_family='BSD',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Hydrology',
        'Topic :: Scientific/Engineering :: Visualization',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
    ],
    python_requires=">=3",
    install_requires=install_requires
)