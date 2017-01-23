cudnnenv
========

cudnnenv manages various versions of cuDNN.


Install
-------

Install cudnnenv via pip command.

::

   $ pip install cudnnenv

Do not forget to set your environment variables.
cuDNN which cudnnenv installs locates at `~/.cudnn/active/cuda`.

::

   LD_LIBRARY_PATH=~/.cudnn/active/cuda/lib64:$LD_LIBRARY_PATH
   CPATH=~/.cudnn/active/cuda/include:$CPATH
   LIBRARY_PATH=~/.cudnn/active/cuda/lib64:$LIBRARY_PATH


Usage
-----

::

   usage: cudnnenv [-h] {install,version,versions,deactivate} ...

positional arguments:
  {install,version,versions,deactivate}

:`install`: Install version
:`uninstall`: Uninstall version
:`version`: Show active version
:`versions`: Show avalable versions
:`deactivate`: Deactivate cudnnenv

optional arguments:
  -h, --help  show this help message and exit


`install`
~~~~~~~~~

`install` subcommand installs a given version of cuDNN and activate it.
You also need to use this command when you want to only activate a version.

::

   usage: cudnnenv install [-h] VERSION

positional arguments:

:`VERSION`: Version of cuDNN you want to install and activate. Select from [v2, v3, v4, v5, v5-cuda8, v51, v51-cuda8]


`uninstall`
~~~~~~~~~~~

`uninstall` subcommand uninstalls a given version of cuDNN from your environment.

::

   usage: cudnnenv uninstall [-h] VERSION

positional arguments:
   
:VERSION: Version of cuDNN you want to uninstall.


`version`
~~~~~~~~~

`version` subcommand shows the current activated version.
If you activate no version, it shows `(none)`.

::

   usage: cudnnenv version [-h]


`versions`
~~~~~~~~~~

`versions` subcommand shows the available versions, which you can select in `install` subcommand.

::

   usage: cudnnenv versions [-h]


`deactivate`
~~~~~~~~~~~~

`deactivate` subcommand deactivates cudnnenv by removing symbolic link.

::

   usage: cudnnenv deactivate [-h]



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
