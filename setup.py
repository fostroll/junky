import os
from setuptools import setup, find_packages

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
version_file_contents = open(os.path.join(SCRIPT_DIR, 'junky/_version.py'),
                             'rt', encoding='utf-8').read()
VERSION = version_file_contents.strip()[len('__version__ = "'):-1]

setup(
    name='junky',
    version=VERSION,
    description='For now, just unsorted utilities for PyTorch',
    long_description=open('README.md', 'rt', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    author='Sergei Ternovykh, Anastasiya Nikiforova',
    author_email='fostroll@gmail.com, ',
    url='https://github.com/fostroll/junky',
    license='BSD',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',
        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Information Technology',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Text Processing',
        'Topic :: Text Processing :: Linguistic',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    # What does your project relate to?
    keywords='pytorch autotrain',

    packages=find_packages(exclude=['doc', 'examples', 'scripts', 'tests']),
    install_requires=['matplotlib', 'numpy', 'pandas', 'seaborn', 'sklearn',
                      'torch>=1.2.0'],
    include_package_data=True,
    python_requires='>=3.5',
)
