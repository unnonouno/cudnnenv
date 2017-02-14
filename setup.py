#!/usr/bin/env python

from setuptools import setup


setup(
    name='cudnnenv',
    version='0.3.0',
    description='cudnn environment manager',
    long_description=open('README.rst').read(),
    author='Yuya Unno',
    author_email='unnonouno@gmail.com',
    url='https://github.com/unnonouno/cudnnenv',
    license='MIT License',
    packages=['cudnnenv'],
    entry_points={
        'console_scripts': ['cudnnenv=cudnnenv:main'],
    },
    classifiers=[
        'Operating System :: POSIX :: Linux',
        'Environment :: Console',
        'Programming Language :: Python',
        'License :: OSI Approved :: MIT License',
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Information Technology',
        'Topic :: Utilities',
    ],
)
