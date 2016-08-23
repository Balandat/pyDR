import os
from setuptools import setup


def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname)) as f:
        return f.read()

reqs = ['numpy>=1.7.1', 'scipy>=0.12', 'pandas>=0.18', 'pytz', 'patsy>=0.4.1']
# we need logutils for python<3.2
reqs += ['logutils>=0.3.3']

# add data files to package
data_dir = os.path.join('pyDR', 'data')
data_files = [(root, [os.path.join(root, f) for f in files])
              for root, dirs, files in os.walk(data_dir)]

setup(
    name='pyDR',
    version='0.1a',
    packages=['pyDR'],
    install_requires=reqs,
    data_files=data_files,
    author='Maximilian Balandat',
    author_email='balandat@eecss.berkeley.edu',
    description='A package for simulating behavior of consumers ' +
                'facing dynamic electricity pricing.',
    license='MIT',
    keywords='economics, electricity, Demand Response',
    url='http://github.com/Balandat/pyDR',
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
    ],
)
