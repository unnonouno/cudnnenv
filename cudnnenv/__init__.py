import argparse
import contextlib
import os
import shutil
import subprocess
import sys
import tempfile

__version__ = '0.2.0'


cudnn_home = os.path.join(os.environ['HOME'], '.cudnn')

codes = {}

cudnn2_base = '''
curl -s -o {cudnn}.tgz http://developer.download.nvidia.com/compute/redist/cudnn/{cudnn_ver}/{cudnn}.tgz &&
echo "{sha256sum}  {cudnn}.tgz" | sha256sum -cw --quiet - &&
tar -xzf {cudnn}.tgz &&
rm {cudnn}.tgz &&
mkdir -p {{path}}/cuda/include &&
mkdir -p {{path}}/cuda/lib64 &&
mv {cudnn}/cudnn.h {{path}}/cuda/include/. &&
mv {cudnn}/libcudnn.so {{path}}/cuda/lib64/. &&
mv {cudnn}/libcudnn.so.6.5 {{path}}/cuda/lib64/. &&
mv {cudnn}/libcudnn.so.6.5.48 {{path}}/cuda/lib64/. &&
mv {cudnn}/libcudnn_static.a {{path}}/cuda/lib64/.
'''

codes['v2'] = cudnn2_base.format(
    cudnn='cudnn-6.5-linux-x64-v2',
    cudnn_ver='v2',
    sha256sum='4b02cb6bf9dfa57f63bfff33e532f53e2c5a12f9f1a1b46e980e626a55f380aa',
)

cudnn_base = '''
curl -o {cudnn}.tgz http://developer.download.nvidia.com/compute/redist/cudnn/{cudnn_ver}/{cudnn}.tgz &&
echo "{sha256sum} {cudnn}.tgz" | sha256sum -cw --quiet - &&
tar -xzf {cudnn}.tgz -C {{path}} &&
rm {cudnn}.tgz
'''

codes['v3'] = cudnn_base.format(
    cudnn='cudnn-7.0-linux-x64-v3.0-prod',
    cudnn_ver='v3',
    sha256sum='98679d5ec039acfd4d81b8bfdc6a6352d6439e921523ff9909d364e706275c2b',
)

codes['v4'] = cudnn_base.format(
    cudnn='cudnn-7.0-linux-x64-v4.0-prod',
    cudnn_ver='v4',
    sha256sum='4e64ef7716f20c87854b4421863328e17cce633330c319b5e13809b61a36f97d',
)

codes['v5'] = cudnn_base.format(
    cudnn='cudnn-7.5-linux-x64-v5.0-ga',
    cudnn_ver='v5',
    sha256sum='c4739a00608c3b66a004a74fc8e721848f9112c5cb15f730c1be4964b3a23b3a',
)

codes['v5-cuda8'] = cudnn_base.format(
    cudnn='cudnn-8.0-linux-x64-v5.0-ga',
    cudnn_ver='v5',
    sha256sum='af80eb1ce0cb51e6a734b2bdc599e6d50b676eab3921e5bddfe5443485df86b6',
)

codes['v51'] = cudnn_base.format(
    cudnn='cudnn-7.5-linux-x64-v5.1',
    cudnn_ver='v5.1',
    sha256sum='69ca71f7728b54b6e003393083f419b24774fecd3b08bbf41bceac9a9fe16345',
)

codes['v51-cuda8'] = cudnn_base.format(
    cudnn='cudnn-8.0-linux-x64-v5.1',
    cudnn_ver='v5.1',
    sha256sum='c10719b36f2dd6e9ddc63e3189affaa1a94d7d027e63b71c3f64d449ab0645ce',
)


@contextlib.contextmanager
def safe_temp_dir():
    temp_dir = tempfile.mkdtemp()
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@contextlib.contextmanager
def safe_dir(path):
    os.makedirs(path)
    try:
        yield path
    except:
        shutil.rmtree(path, ignore_errors=True)
        raise


def get_version_path(ver):
    return os.path.join(cudnn_home, 'versions', ver)


def get_active_path():
    return os.path.join(cudnn_home, 'active')


def download_cudnn(ver):
    path = get_version_path(ver)

    with safe_dir(path), safe_temp_dir() as temp_dir:
        os.chdir(temp_dir)
        cmd = codes[ver].format(path=path)
        subprocess.check_call(cmd, shell=True)


def ensure_exist(ver):
    path = get_version_path(ver)
    if not os.path.exists(path):
        download_cudnn(ver)


def remove_link():
    symlink_path = get_active_path()
    if os.path.exists(symlink_path):
        os.remove(symlink_path)


def select_cudnn(ver):
    ensure_exist(ver)

    version_path = os.path.join('versions', ver)
    remove_link()
    symlink_path = get_active_path()
    os.symlink(version_path, symlink_path)


def yes_no_query(question):
    while True:
        user_input = raw_input('%s [y/n] ' % question).lower()
        if user_input == 'y':
            return True
        elif user_input == 'n':
            return False


def uninstall_cudnn(ver):
    path = get_version_path(ver)
    if not os.path.exists(path):
        print('version %s is not installed' % ver)
        sys.exit(2)

    if yes_no_query('remove %s?' % path):
        shutil.rmtree(path, ignore_errors=True)


def install(args):
    select_cudnn(args.version)


def uninstall(args):
    uninstall_cudnn(args.version)


def get_version():
    symlink_path = get_active_path()
    if os.path.islink(symlink_path):
        path = os.readlink(symlink_path)
        return os.path.split(path)[-1]
    else:
        return None


def version(args):
    ver = get_version()
    if ver is None:
        print('(none)')
    else:
        print(ver)


def versions(args):
    active = get_version()
    for ver in sorted(codes.keys()):
        if ver == active:
            ver = '* ' + ver
        else:
            ver = '  ' + ver
        print(ver)


def deactivate(args):
    remove_link()


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='Subcommand')

    sub = subparsers.add_parser('install', help='Install version')
    vers = sorted(codes.keys())
    sub.add_argument(
        'version', metavar='VERSION', choices=vers,
        help='Version of cuDNN you want to install and activate. '
        'Select from [%s]' % ', '.join(vers))
    sub.set_defaults(func=install)

    sub = subparsers.add_parser('uninstall', help='Uninstall version')
    sub.add_argument(
        'version', metavar='VERSION',
        help='Version of cuDNN you want to uninstall.')
    sub.set_defaults(func=uninstall)

    sub = subparsers.add_parser('version', help='Show active version')
    sub.set_defaults(func=version)

    sub = subparsers.add_parser('versions', help='Show avalable versions')
    sub.set_defaults(func=versions)

    sub = subparsers.add_parser('deactivate', help='Deactivate cudnnenv')
    sub.set_defaults(func=deactivate)

    args = parser.parse_args()

    if not hasattr(args, 'func'):
        parser.error('specify sub-command')
        sys.exit(1)

    args.func(args)
