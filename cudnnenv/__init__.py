from __future__ import print_function
from __future__ import unicode_literals

import argparse
import contextlib
import os
import platform
import shutil
import subprocess
import sys
import tempfile

__version__ = '0.6.5'


if int(platform.python_version_tuple()[0]) >= 3:
    _raw_input = input
else:
    _raw_input = raw_input  # NOQA


cudnn_home = os.path.join(os.environ['HOME'], '.cudnn')

codes = {}

if 'linux' in sys.platform:
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
echo "{sha256sum}  {cudnn}.tgz" | sha256sum -cw --quiet - &&
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
        sha256sum='cd091763d5889f0efff1fbda83bade191f530743a212c6b0ecc2a64d64d94405',
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

    codes['v6'] = cudnn_base.format(
        cudnn='cudnn-7.5-linux-x64-v6.0',
        cudnn_ver='v6.0',
        sha256sum='568d4b070c5f91ab8a15b287b73dd072b99c7267a43edad13f70337cd186c82c',
    )

    codes['v6-cuda8'] = cudnn_base.format(
        cudnn='cudnn-8.0-linux-x64-v6.0',
        cudnn_ver='v6.0',
        sha256sum='9b09110af48c9a4d7b6344eb4b3e344daa84987ed6177d5c44319732f3bb7f9c',
    )

    codes['v7.0.1-cuda8'] = cudnn_base.format(
        cudnn='cudnn-8.0-linux-x64-v7',
        cudnn_ver='v7.0.1',
        sha256sum='288d844ab289b56d0c7b6719a34c7c0b57a01c58ffbe4d582c9b539da96ed2a7',
    )

    codes['v7.0.1-cuda9'] = cudnn_base.format(
        cudnn='cudnn-9.0-linux-x64-v7',
        cudnn_ver='v7.0.1',
        sha256sum='32d0caf6250ea8d2c3c80649ea6a032e46741d78bdca40c37b8ac67b00fe3244',
    )

    codes['v7.0.2-cuda8'] = cudnn_base.format(
        cudnn='cudnn-8.0-linux-x64-v7',
        cudnn_ver='v7.0.2',
        sha256sum='b667807f2b82af7a9ed5451e9ff5ea7a11deeef85aafdc5529e1adfddcc069ca',
    )

    codes['v7.0.2-cuda9'] = cudnn_base.format(
        cudnn='cudnn-9.0-linux-x64-v7',
        cudnn_ver='v7.0.2',
        sha256sum='ec2a89453ef6454d417b7f3dad67405e30953e1df1e47aafb846f99d02eaa5d1',
    )

    codes['v7.0.3-cuda8'] = cudnn_base.format(
        cudnn='cudnn-8.0-linux-x64-v7',
        cudnn_ver='v7.0.3',
        sha256sum='e44dd20750ea1fa287ed0576c71d7ba30383aabdaacd18df173947cf7a53fc3f',
    )

    codes['v7.0.3-cuda9'] = cudnn_base.format(
        cudnn='cudnn-9.0-linux-x64-v7',
        cudnn_ver='v7.0.3',
        sha256sum='09583e93110cee2bf76ea355e1d9c7c366a50ad858362064f7c927cc46209ef9',
    )

    codes['v7.0.4-cuda8'] = cudnn_base.format(
        cudnn='cudnn-8.0-linux-x64-v7',
        cudnn_ver='v7.0.4',
        sha256sum='c9d6e482063407edaa799c944279e5a1a3a27fd75534982076e62b1bebb4af48',
    )

    codes['v7.0.4-cuda9'] = cudnn_base.format(
        cudnn='cudnn-9.0-linux-x64-v7',
        cudnn_ver='v7.0.4',
        sha256sum='963da2057c298616dab0c754398dcb1cced1bdc5f21ca00258b149f709b5bc4f',
    )

    codes['v7.0.5-cuda8'] = cudnn_base.format(
        cudnn='cudnn-8.0-linux-x64-v7',
        cudnn_ver='v7.0.5',
        sha256sum='9e0b31735918fe33a79c4b3e612143d33f48f61c095a3b993023cdab46f6d66e',
    )

    codes['v7.0.5-cuda9'] = cudnn_base.format(
        cudnn='cudnn-9.0-linux-x64-v7',
        cudnn_ver='v7.0.5',
        sha256sum='1a3e076447d5b9860c73d9bebe7087ffcb7b0c8814fd1e506096435a2ad9ab0e',
    )

    codes['v7.0.5-cuda91'] = cudnn_base.format(
        cudnn='cudnn-9.1-linux-x64-v7',
        cudnn_ver='v7.0.5',
        sha256sum='1ead5da7324db35dcdb3721a8d4fc020b217c68cdb3b3daa1be81eb2456bd5e5',
    )

    codes['v7-cuda8'] = codes['v7.0.5-cuda8']
    codes['v7-cuda9'] = codes['v7.0.5-cuda9']
    codes['v7-cuda91'] = codes['v7.0.5-cuda91']

    codes['v7.1.1-cuda8'] = cudnn_base.format(
        cudnn='cudnn-8.0-linux-x64-v7.1',
        cudnn_ver='v7.1.1',
        sha256sum='4a9d59ecee53e830f24e90d689dbab1aa9f69efced823f33046040901c4151e1',
    )

    codes['v7.1.1-cuda9'] = cudnn_base.format(
        cudnn='cudnn-9.0-linux-x64-v7.1',
        cudnn_ver='v7.1.1',
        sha256sum='d2038dca6e6070aa6879d827fa6c032c942514a6b9bddf5ade275670ca474b9c',
    )

    codes['v7.1.1-cuda91'] = cudnn_base.format(
        cudnn='cudnn-9.1-linux-x64-v7.1',
        cudnn_ver='v7.1.1',
        sha256sum='ae3cf4f6d0d5b39c74742dadb44c91e51531b79e3d01a7aab3459ab9bed2f475',
    )

    codes['v7.1.2-cuda8'] = cudnn_base.format(
        cudnn='cudnn-8.0-linux-x64-v7.1',
        cudnn_ver='v7.1.2',
        sha256sum='a5dfd656811fe9c43d87c40338cec5e1a85ad5a9cd3b5f1e95dd67d5e126aacc',
    )

    codes['v7.1.2-cuda9'] = cudnn_base.format(
        cudnn='cudnn-9.0-linux-x64-v7.1',
        cudnn_ver='v7.1.2',
        sha256sum='d5f8b56a05dc002a801f8f2081518c20b496006f14617d568bde24dcfb3f6388',
    )

    codes['v7.1.2-cuda91'] = cudnn_base.format(
        cudnn='cudnn-9.1-linux-x64-v7.1',
        cudnn_ver='v7.1.2',
        sha256sum='c61000ed700bc5a009bc2e135bbdf736c9743212b2174a2fc9018a66cc0979ec',
    )

    codes['v7.1.3-cuda8'] = cudnn_base.format(
        cudnn='cudnn-8.0-linux-x64-v7.1',
        cudnn_ver='v7.1.3',
        sha256sum='31ed3c3bfb9c515c228c1dcbb306277ce08836e84e3facedef6182d872f8cd3d',
    )

    codes['v7.1.3-cuda9'] = cudnn_base.format(
        cudnn='cudnn-9.0-linux-x64-v7.1',
        cudnn_ver='v7.1.3',
        sha256sum='203a1700cc5b96f679d550d8bbcf99bdc254654b399e6bf79ef8ed6e6b83a369',
    )

    codes['v7.1.3-cuda91'] = cudnn_base.format(
        cudnn='cudnn-9.1-linux-x64-v7.1',
        cudnn_ver='v7.1.3',
        sha256sum='dd616d3794167ceb923d706bf73e8d6acdda770751492b921ee6827cdf190228',
    )

    codes['v7.1.4-cuda9'] = cudnn_base.format(
        cudnn='cudnn-9.0-linux-x64-v7.1',
        cudnn_ver='v7.1.4',
        sha256sum='60b581d0f05324c33323024a264aa3fb185c533e2f67dae7fda847b926bb7e57',
    )

    codes['v7.1.4-cuda92'] = cudnn_base.format(
        cudnn='cudnn-9.2-linux-x64-v7.1',
        cudnn_ver='v7.1.4',
        sha256sum='f875340f812b942408098e4c9807cb4f8bdaea0db7c48613acece10c7c827101',
    )

    codes['v7.2.1-cuda8'] = cudnn_base.format(
        cudnn='cudnn-8.0-linux-x64-v7.2.1.38',
        cudnn_ver='v7.2.1',
        sha256sum='c2d58788fd51d892fb84a1fae578d8cb432f7301b279d0a1cf7b38faf79993f4',
    )

    codes['v7.2.1-cuda9'] = cudnn_base.format(
        cudnn='cudnn-9.0-linux-x64-v7.2.1.38',
        cudnn_ver='v7.2.1',
        sha256sum='cf007437b9ac6250ec63b89c25f248d2597fdd01369c80146567f78e75ce4e37',
    )

    codes['v7.2.1-cuda92'] = cudnn_base.format(
        cudnn='cudnn-9.2-linux-x64-v7.2.1.38',
        cudnn_ver='v7.2.1',
        sha256sum='3e78f5f0edbe614b56f00ff2d859c5409d150c87ae6ba3df09f97d537909c2e9',
    )

    codes['v7.3.0-cuda9'] = cudnn_base.format(
        cudnn='cudnn-9.0-linux-x64-v7.3.0.29',
        cudnn_ver='v7.3.0',
        sha256sum='403f9043ff2c7b2c5967454872275d07bca11fd41dfc7b21995eadcad6dbe49b',
    )

    codes['v7.3.0-cuda10'] = cudnn_base.format(
        cudnn='cudnn-10.0-linux-x64-v7.3.0.29',
        cudnn_ver='v7.3.0',
        sha256sum='7526a33bc3c152ca5d8f3eddedaa4a0b3c721a3c0000eeb80ebfe5cbc54696b7',
    )

    codes['v71-cuda8'] = codes['v7.1.3-cuda8']
    codes['v71-cuda9'] = codes['v7.1.4-cuda9']
    codes['v71-cuda91'] = codes['v7.1.3-cuda91']
    codes['v71-cuda92'] = codes['v7.1.4-cuda92']
    codes['v72-cuda8'] = codes['v7.2.1-cuda8']
    codes['v72-cuda9'] = codes['v7.2.1-cuda9']
    codes['v72-cuda92'] = codes['v7.2.1-cuda92']
    codes['v73-cuda9'] = codes['v7.3.0-cuda9']
    codes['v73-cuda10'] = codes['v7.3.0-cuda10']

    LIBDIR = 'lib64'

elif sys.platform == 'darwin':
    cudnn2_base = '''
curl -s -o {cudnn}.tgz http://developer.download.nvidia.com/compute/redist/cudnn/{cudnn_ver}/{cudnn}.tgz &&
echo "{sha256sum}  {cudnn}.tgz" | shasum -a 256 -c - &&
tar -xzf {cudnn}.tgz &&
rm {cudnn}.tgz &&
mkdir -p {{path}}/cuda/include &&
mkdir -p {{path}}/cuda/lib &&
mv {cudnn}/cudnn.h {{path}}/cuda/include/. &&
mv {cudnn}/libcudnn.dylib {{path}}/cuda/lib/. &&
mv {cudnn}/libcudnn.6.5.dylib {{path}}/cuda/lib/. &&
mv {cudnn}/libcudnn_static.a {{path}}/cuda/lib/.
'''

    codes['v2'] = cudnn2_base.format(
        cudnn='cudnn-6.5-osx-v2',
        cudnn_ver='v2',
        sha256sum='7dde2658e9861bb270c327fb3d806232579d48e77b6f495b26c17a4717af97c1',
    )

    cudnn_base = '''
curl -o {cudnn}.tgz http://developer.download.nvidia.com/compute/redist/cudnn/{cudnn_ver}/{cudnn}.tgz &&
echo "{sha256sum}  {cudnn}.tgz" | shasum -a 256 -c - &&
tar -xzf {cudnn}.tgz -C {{path}} &&
rm {cudnn}.tgz
'''

    codes['v3'] = cudnn_base.format(
        cudnn='cudnn-7.0-osx-x64-v3.0-prod',
        cudnn_ver='v3',
        sha256sum='48cf77784bf0f833f3e52402aa6ff359aaca03e8d4aa48880ef2a00d91693633',
    )

    codes['v4'] = cudnn_base.format(
        cudnn='cudnn-7.0-osx-x64-v4.0-prod',
        cudnn_ver='v4',
        sha256sum='675ed2bebe67fe317306fae3c44024ec9a848ee5b0e1fdcb14f9882a4d91aa4e',
    )

    codes['v5'] = cudnn_base.format(
        cudnn='cudnn-7.5-osx-x64-v5.0-ga',
        cudnn_ver='v5',
        sha256sum='3008aa04b599650493c80daad802d893485af258fd503380e6fd5fa4569a3a73',
    )

    codes['v5-cuda8'] = cudnn_base.format(
        cudnn='cudnn-8.0-osx-x64-v5.0-ga',
        cudnn_ver='v5',
        sha256sum='25dea96077c1d90ba4cb0a34e7d6dc8e885b8d1437e4ce8c7bce80f66ca10252',
    )

    codes['v51'] = cudnn_base.format(
        cudnn='cudnn-7.5-osx-x64-v5.1',
        cudnn_ver='v5.1',
        sha256sum='bfea8f20351fc5d6ecd26b55e5c06850726f68b3df2f9c74d9f4ec77cb467f89',
    )

    codes['v51-cuda8'] = cudnn_base.format(
        cudnn='cudnn-8.0-osx-x64-v5.1',
        cudnn_ver='v5.1',
        sha256sum='2528f09bfbfafc7397682308d0809f22609b57eea9faec51f320ce8aab2ebec6',
    )

    codes['v6'] = cudnn_base.format(
        cudnn='cudnn-7.5-osx-x64-v6.0',
        cudnn_ver='v6.0',
        sha256sum='368fd7e197d735e84446d97da3e27c1c22934dfc960cca92a22160200a2f6d17',
    )

    codes['v6-cuda8'] = cudnn_base.format(
        cudnn='cudnn-8.0-osx-x64-v6.0',
        cudnn_ver='v6.0',
        sha256sum='1f2842872ddae3ca61d5f5a89fe5f519f18ca538c28cc04c0c4bb3b5f3317dac'
    )

    codes['v7.0.1-cuda9'] = cudnn_base.format(
        cudnn='cudnn-9.0-osx-x64-v7',
        cudnn_ver='v7.0.1',
        sha256sum='b2c259fc6b47abd820397e37194b49ff538f03c07d0121a66475986e61d11c5d',
    )

    codes['v7.0.2-cuda9'] = cudnn_base.format(
        cudnn='cudnn-9.0-osx-x64-v7',
        cudnn_ver='v7.0.2',
        sha256sum='3c9dd6d17aad84678934c735f4a1a5a901a2c4c79cc45b3e2cb5599834fb1a2b',
    )

    codes['v7.0.3-cuda9'] = cudnn_base.format(
        cudnn='cudnn-9.0-osx-x64-v7',
        cudnn_ver='v7.0.3',
        sha256sum='ea7e085af13de736e2727a21d2cd0162084afc12b17fdb08b124d0e5280bab11',
    )

    codes['v7.1.4-cuda92'] = cudnn_base.format(
        cudnn='cudnn-9.2-osx-x64-v7.1',
        cudnn_ver='v7.1.4',
        sha256sum='86d1354462cc893fafb6d0e04994ff3dcda5791ffef6b11de0f0f51d0c9ac7fc',
    )

    codes['v7.2.1-cuda92'] = cudnn_base.format(
        cudnn='cudnn-9.2-osx-x64-v7.2.1.38',
        cudnn_ver='v7.2.1',
        sha256sum='62f92b4b70fa876715fcef062776046f9d2a4ff11d0b6d6eabddefa48626d34f',
    )

    codes['v7.3.0-cuda10'] = cudnn_base.format(
        cudnn='cudnn-10.0-osx-x64-v7.3.0.29',
        cudnn_ver='v7.3.0',
        sha256sum='a722448a4efa2448028543fdb563ea1b8c3bb8a99da53d041e668b1dfdc76099',
    )

    codes['v7-cuda9'] = codes['v7.0.3-cuda9']
    codes['v71-cuda92'] = codes['v7.1.4-cuda92']
    codes['v72-cuda92'] = codes['v7.2.1-cuda92']
    codes['v73-cuda10'] = codes['v7.3.0-cuda10']

    LIBDIR = 'lib'

else:
    print('Unsupported platform: "%s"' % sys.platform)
    sys.exit(1)


local_install_command = 'tar -xzf {file} -C {path}'


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
    except BaseException:
        shutil.rmtree(path, ignore_errors=True)
        raise


def get_version_path(ver):
    return os.path.join(cudnn_home, 'versions', ver)


def get_active_path():
    return os.path.join(cudnn_home, 'active')


def get_installed_versions():
    version_dir = os.path.join(cudnn_home, 'versions')
    if not os.path.isdir(version_dir):
        return []
    return os.listdir(version_dir)


def download_cudnn(ver):
    path = get_version_path(ver)

    with safe_dir(path), safe_temp_dir() as temp_dir:
        os.chdir(temp_dir)
        cmd = codes[ver].format(path=path)
        subprocess.check_call(cmd, shell=True)


def download_if_not_exist(ver):
    path = get_version_path(ver)
    if not os.path.exists(path):
        download_cudnn(ver)


def ensure_exist(ver):
    path = get_version_path(ver)
    if not os.path.exists(path):
        print('version %s is not installed' % ver)
        sys.exit(2)


def remove_link():
    symlink_path = get_active_path()
    if os.path.lexists(symlink_path):
        os.remove(symlink_path)


def select_cudnn(ver):
    ensure_exist(ver)

    version_path = os.path.join('versions', ver)
    remove_link()
    symlink_path = get_active_path()
    os.symlink(version_path, symlink_path)
    print('Successfully installed %s' % ver)
    print('Set your environment variables:')
    print('')
    print('  LD_LIBRARY_PATH=~/.cudnn/active/cuda/%s:$LD_LIBRARY_PATH' % LIBDIR)
    print('  CPATH=~/.cudnn/active/cuda/include:$CPATH')
    print('  LIBRARY_PATH=~/.cudnn/active/cuda/%s:$LIBRARY_PATH' % LIBDIR)


def yes_no_query(question):
    while True:
        user_input = _raw_input('%s [y/n] ' % question).lower()
        if user_input == 'y':
            return True
        elif user_input == 'n':
            return False


def uninstall_cudnn(ver):
    ensure_exist(ver)

    path = get_version_path(ver)
    if yes_no_query('remove %s?' % path):
        shutil.rmtree(path, ignore_errors=True)


def install(args):
    download_if_not_exist(args.version)
    select_cudnn(args.version)


def activate(args):
    select_cudnn(args.version)


def install_file(args):
    path = get_version_path(args.version)
    if os.path.exists(path):
        print('version %s already exists' % args.version)
        sys.exit(3)

    path = get_version_path(args.version)
    with safe_dir(path):
        cmd = local_install_command.format(
            file=args.file, path=path)
        subprocess.check_call(cmd, shell=True)

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


def print_versions(versions, active):
    for ver in sorted(versions):
        if ver == active:
            ver = '* ' + ver
        else:
            ver = '  ' + ver
        print(ver)


def version(args):
    ver = get_version()
    if ver is None:
        print('(none)')
    else:
        print(ver)


def versions(args):
    active = get_version()
    print('Available versions:')
    print_versions(codes.keys(), active)
    print('')
    print('Installed versions:')
    print_versions(get_installed_versions(), active)


def deactivate(args):
    remove_link()


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--version', action='version', version='cudnnenv %s' % __version__)
    subparsers = parser.add_subparsers(help='Subcommand')

    sub = subparsers.add_parser('install', help='Install version')
    vers = sorted(codes.keys())
    sub.add_argument(
        'version', metavar='VERSION', choices=vers,
        help='Version of cuDNN you want to install and activate. '
        'Select from [%s]' % ', '.join(vers))
    sub.set_defaults(func=install)

    sub = subparsers.add_parser('install-file', help='Install local cuDNN file')
    sub.add_argument(
        'file', metavar='FILE',
        help='Path to local cuDNN archive file to install')
    sub.add_argument(
        'version', metavar='VERSION',
        help='Version name of cuDNN you want to install')
    sub.set_defaults(func=install_file)

    sub = subparsers.add_parser('activate', help='Activate installed version')
    vers = sorted(codes.keys())
    sub.add_argument(
        'version', metavar='VERSION',
        help='Version of installed cuDNN you want to activate. ')
    sub.set_defaults(func=activate)

    sub = subparsers.add_parser('uninstall', help='Uninstall version')
    sub.add_argument(
        'version', metavar='VERSION',
        help='Version of cuDNN you want to uninstall.')
    sub.set_defaults(func=uninstall)

    sub = subparsers.add_parser('version', help='Show active version')
    sub.set_defaults(func=version)

    sub = subparsers.add_parser('versions', help='Show available versions')
    sub.set_defaults(func=versions)

    sub = subparsers.add_parser('deactivate', help='Deactivate cudnnenv')
    sub.set_defaults(func=deactivate)

    args = parser.parse_args(args=args)

    if not hasattr(args, 'func'):
        parser.error('too few arguments')

    args.func(args)
