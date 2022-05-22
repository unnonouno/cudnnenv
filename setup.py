#!/usr/bin/env python

from setuptools import setup


setup(
    name='cudnnenv',
    version='0.8.0',
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
        'Operating System :: MacOS :: MacOS X',
        'Environment :: Console',
        'Programming Language :: Python',
        'License :: OSI Approved :: MIT License',
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Information Technology',
        'Topic :: Utilities',
    ],
    tests_require=['mock<=2.0.0'],
    test_suite='test',
)
