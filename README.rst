cudnnenv
========

cudnnenv manages various versions of cuDNN.


Install
-------

::

   LD_LIBRARY_PATH=~/.cudnn/active/cuda/lib64:$LD_LIBRARY_PATH
   CPATH=~/.cudnn/active/cuda/include:$CPATH
   LIBRARY_PATH=~/.cudnn/active/cuda/lib64:$LIBRARY_PATH


Directory structure
-------------------

::

  + .cudnn
    + versions
    | + v2
    | | + cuda
    | |   + include
    | |   + lib64
    | + v3
    | + ...
    + active --> versions/vX


License
-------

cudnnenv is distributed under MIT License.
