.. image:: https://img.shields.io/pypi/v/cudnnenv.svg
   :target: https://pypi.python.org/pypi/cudnnenv

.. image:: https://img.shields.io/github/license/unnonouno/cudnnenv.svg
   :target: https://github.com/unnonouno/cudnnenv

.. image:: https://img.shields.io/travis/unnonouno/cudnnenv/master.svg
   :target: https://travis-ci.org/unnonouno/cudnnenv

.. image:: https://img.shields.io/coveralls/unnonouno/cudnnenv.svg
   :target: https://coveralls.io/r/unnonouno/cudnnenv?branch=master


cudnnenv
========

cudnnenv manages various versions of cuDNN.


Requirement
-----------

- Linux, macOS
- Python 2.7, 3.4, 3.5, 3.6


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

This program uses ``curl`` and ``tar`` commands.
Please install them before you use it.


Usage
-----

::

   usage: cudnnenv [-h]
                   {install,install-file,activate,uninstall,version,versions,deactivate}
                   ...

positional arguments:
  {install,install-file,activate,uninstall,version,versions,deactivate}

:`install`: Install version
:`install-file`: Install local cuDNN file
:`activate`: Activate installed version
:`uninstall`: Uninstall version
:`version`: Show active version
:`versions`: Show avalable versions
:`deactivate`: Deactivate cudnnenv

optional arguments:
  -h, --help  show this help message and exit


`install`
~~~~~~~~~

`install` subcommand installs a given version of cuDNN and activate it.
Use `activate` subcommand to only activate installed version.

::

   usage: cudnnenv install [-h] VERSION

positional arguments:

:`VERSION`: Version of cuDNN you want to install and activate. Select from [v2, v3, v4, v5, v5-cuda8, v51, v51-cuda8, v6, v6-cuda8, v7-cuda8, v7-cuda9]

Note that `v7-cuda8` is not available on macOS.

`install-file`
~~~~~~~~~~~~~~

`install-file` subcommand installs a given local cuDNN file and activate it.
You can only install tar.gz file, can not use deb packages.

::

   usage: cudnnenv install-file [-h] FILE VERSION

positional arguments:

:`FILE`: Path to local cuDNN archive file to install
:`VERSION`: Version name of cuDNN you want to install


`activate`
~~~~~~~~~~

`activate` subcommand activates an installed cuDNN.
This command does not download an archive file unlike `install`.

::

   usage: cudnnenv activate [-h] VERSION

positional arguments:

:`VERSION`: Version of installed cuDNN you want to activate.


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

`versions` subcommand shows the available versions which you can select in `install` subcommand, and installed versions which you installed with `install` and `install-files` subcommands.

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
