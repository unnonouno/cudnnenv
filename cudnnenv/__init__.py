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

__version__ = '0.8.0'


if int(platform.python_version_tuple()[0]) >= 3:
    _raw_input = input
else:
    _raw_input = raw_input  # NOQA


cudnn_home = os.path.join(os.environ['HOME'], '.cudnn')

codes = {}

if 'linux' in sys.platform:
    cudnn2_base = '''
curl -s -o {cudnn}.tgz https://developer.download.nvidia.com/compute/redist/cudnn/{cudnn_ver}/{cudnn}.tgz &&
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
curl -o {cudnn}.tgz https://developer.download.nvidia.com/compute/redist/cudnn/{cudnn_ver}/{cudnn}.tgz &&
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

    codes['v7.4.1-cuda9'] = cudnn_base.format(
        cudnn='cudnn-9.0-linux-x64-v7.4.1.5',
        cudnn_ver='v7.4.1',
        sha256sum='bec38fc281fec0226766cce050473043765345cb8a5ed699da4d663ecfa4f24d',
    )

    codes['v7.4.1-cuda92'] = cudnn_base.format(
        cudnn='cudnn-9.2-linux-x64-v7.4.1.5',
        cudnn_ver='v7.4.1',
        sha256sum='a850d62f32c6a18271932d9a96072ac757c2c516bd1200ae8b79e4bdd3800b5b',
    )

    codes['v7.4.1-cuda10'] = cudnn_base.format(
        cudnn='cudnn-10.0-linux-x64-v7.4.1.5',
        cudnn_ver='v7.4.1',
        sha256sum='b320606f1840eec0cdd4453cb333554a3fe496dd4785f10d8e87fe1a4f52bd5c',
    )

    codes['v7.4.2-cuda9'] = cudnn_base.format(
        cudnn='cudnn-9.0-linux-x64-v7.4.2.24',
        cudnn_ver='v7.4.2',
        sha256sum='e3e72e9e2bf4c5e4cdd467aa6b824effc4566d230a2cda4153ad894d7d15cf73',
    )

    codes['v7.4.2-cuda92'] = cudnn_base.format(
        cudnn='cudnn-9.2-linux-x64-v7.4.2.24',
        cudnn_ver='v7.4.2',
        sha256sum='19565be5dba39097d59f99227fd65cd2f3a3be9e4249500f772d4b14c7806371',
    )

    codes['v7.4.2-cuda10'] = cudnn_base.format(
        cudnn='cudnn-10.0-linux-x64-v7.4.2.24',
        cudnn_ver='v7.4.2',
        sha256sum='2edfc86a02b50d17e88c478955a332e6a1e8174e7e53a3458b4ea51faf02daa3',
    )

    codes['v7.5.0-cuda9'] = cudnn_base.format(
        cudnn='cudnn-9.0-linux-x64-v7.5.0.56',
        cudnn_ver='v7.5.0',
        sha256sum='ee0ecd3cc30b9bf5ec875eac3ed375d3996bcb0ed5d2551716e4884b3ea5ce8c',
    )

    codes['v7.5.0-cuda92'] = cudnn_base.format(
        cudnn='cudnn-9.2-linux-x64-v7.5.0.56',
        cudnn_ver='v7.5.0',
        sha256sum='2a04fd5ed5b8d32e2401c85a1a38f3cfd6da662c31bd26e80bea25469e48a675',
    )

    codes['v7.5.0-cuda10'] = cudnn_base.format(
        cudnn='cudnn-10.0-linux-x64-v7.5.0.56',
        cudnn_ver='v7.5.0',
        sha256sum='701097882cb745d4683bb7ff6c33b8a35c7c81be31bac78f05bad130e7e0b781',
    )

    codes['v7.5.0-cuda101'] = cudnn_base.format(
        cudnn='cudnn-10.1-linux-x64-v7.5.0.56',
        cudnn_ver='v7.5.0',
        sha256sum='c31697d6b71afe62838ad2e57da3c3c9419c4e9f5635d14b683ebe63f904fbc8',
    )

    codes['v7.5.1-cuda9'] = cudnn_base.format(
        cudnn='cudnn-9.0-linux-x64-v7.5.1.10',
        cudnn_ver='v7.5.1',
        sha256sum='1abe08839dcb59a3a7293c85f642bf0dd2486e377d0fbca1b0311f38e183251a',
    )

    codes['v7.5.1-cuda92'] = cudnn_base.format(
        cudnn='cudnn-9.2-linux-x64-v7.5.1.10',
        cudnn_ver='v7.5.1',
        sha256sum='e840d29ce5f0c068911966e62128397c6a9bb5e2ea9c66394a592c3b61e770a5',
    )

    codes['v7.5.1-cuda10'] = cudnn_base.format(
        cudnn='cudnn-10.0-linux-x64-v7.5.1.10',
        cudnn_ver='v7.5.1',
        sha256sum='c0a4ec438920aa581dd567117b9c316745b4a451ac739b1e04939a3d8b229985',
    )

    codes['v7.5.1-cuda101'] = cudnn_base.format(
        cudnn='cudnn-10.1-linux-x64-v7.5.1.10',
        cudnn_ver='v7.5.1',
        sha256sum='2c833f43c9147d9a25a20947a4c5a5f5c33b2443240fd767f63b330c482e68e0',
    )

    codes['v7.6.0-cuda9'] = cudnn_base.format(
        cudnn='cudnn-9.0-linux-x64-v7.6.0.64',
        cudnn_ver='v7.6.0',
        sha256sum='90659ea77734b7b671afe930c9898d21a13b888998f1dd3940cc57d6b2f29b86',
    )

    codes['v7.6.0-cuda92'] = cudnn_base.format(
        cudnn='cudnn-9.2-linux-x64-v7.6.0.64',
        cudnn_ver='v7.6.0',
        sha256sum='ff028e6f07349445c16fef704a90bccb0992c3e012bba66ab1da352bad55b304',
    )

    codes['v7.6.0-cuda10'] = cudnn_base.format(
        cudnn='cudnn-10.0-linux-x64-v7.6.0.64',
        cudnn_ver='v7.6.0',
        sha256sum='c4e1ee4168f4cadabaa989487a47bed09f34d34e35398b6084a2699d11bd2560',
    )

    codes['v7.6.0-cuda101'] = cudnn_base.format(
        cudnn='cudnn-10.1-linux-x64-v7.6.0.64',
        cudnn_ver='v7.6.0',
        sha256sum='e956c6f9222fcb867a10449cfc76dee5cfd7c7531021d95fe9586d7e043b57d7',
    )

    codes['v7.6.1-cuda9'] = cudnn_base.format(
        cudnn='cudnn-9.0-linux-x64-v7.6.1.34',
        cudnn_ver='v7.6.1',
        sha256sum='57a921b20be1d3d1192e59ecfdee61c55e06e530a6d414a013083fa81a77f1c5',
    )

    codes['v7.6.1-cuda92'] = cudnn_base.format(
        cudnn='cudnn-9.2-linux-x64-v7.6.1.34',
        cudnn_ver='v7.6.1',
        sha256sum='0d38735b06a1daf518c7ad4977fdb987a470f7793d95542ac9897e214ab1b006',
    )

    codes['v7.6.1-cuda10'] = cudnn_base.format(
        cudnn='cudnn-10.0-linux-x64-v7.6.1.34',
        cudnn_ver='v7.6.1',
        sha256sum='af0791cba08468a4bf2a4ef15a787dac261f41219caaf335984d47d11eca19ba',
    )

    codes['v7.6.1-cuda101'] = cudnn_base.format(
        cudnn='cudnn-10.1-linux-x64-v7.6.1.34',
        cudnn_ver='v7.6.1',
        sha256sum='9885e38e71fa9844b3e4fb7c7211af41b24c9f76a9014f9d5e1768ddff2087dc',
    )

    codes['v7.6.2-cuda9'] = cudnn_base.format(
        cudnn='cudnn-9.0-linux-x64-v7.6.2.24',
        cudnn_ver='v7.6.2',
        sha256sum='787adab5287597bf9a6462be4bcd55904593690315a889884d8ccfa8aaa9e370',
    )

    codes['v7.6.2-cuda92'] = cudnn_base.format(
        cudnn='cudnn-9.2-linux-x64-v7.6.2.24',
        cudnn_ver='v7.6.2',
        sha256sum='9a015dbda1caa904433c71c89f7d61559c9fa5667ad53663c529efae49966e46',
    )

    codes['v7.6.2-cuda10'] = cudnn_base.format(
        cudnn='cudnn-10.0-linux-x64-v7.6.2.24',
        cudnn_ver='v7.6.2',
        sha256sum='86d87c426537a55dbdbe416b92f8afa1e49361a759c3d17f119e431767a88432',
    )

    codes['v7.6.2-cuda101'] = cudnn_base.format(
        cudnn='cudnn-10.1-linux-x64-v7.6.2.24',
        cudnn_ver='v7.6.2',
        sha256sum='afbfd6a61e774beb3851742452c007de4f65f8ec0592d583bc6806f8d386cd1f',
    )

    codes['v7.6.3-cuda9'] = cudnn_base.format(
        cudnn='cudnn-9.0-linux-x64-v7.6.3.30',
        cudnn_ver='v7.6.3',
        sha256sum='00b4664a36fca4778ed245fe766a13ececef94797720e15e24efcde02fc4c230',
    )

    codes['v7.6.3-cuda92'] = cudnn_base.format(
        cudnn='cudnn-9.2-linux-x64-v7.6.3.30',
        cudnn_ver='v7.6.3',
        sha256sum='d72d276d2e15cbf443fa1f1041a6457a0b238cb321eee36fe80638791f059007',
    )

    codes['v7.6.3-cuda10'] = cudnn_base.format(
        cudnn='cudnn-10.0-linux-x64-v7.6.3.30',
        cudnn_ver='v7.6.3',
        sha256sum='e19e156a10f6ccd57e3323cdb38290ac7c786907c669283130dc2b8a47fbf83b',
    )

    codes['v7.6.3-cuda101'] = cudnn_base.format(
        cudnn='cudnn-10.1-linux-x64-v7.6.3.30',
        cudnn_ver='v7.6.3',
        sha256sum='352557346d8111e2f954c494be1a90207103d316b8777c33e62b3a7f7b708961',
    )

    codes['v7.6.4-cuda9'] = cudnn_base.format(
        cudnn='cudnn-9.0-linux-x64-v7.6.4.38',
        cudnn_ver='v7.6.4',
        sha256sum='8db78c3623c192d4f03f3087b41c32cb0baac95e13408b5d9dabe626cb4aab5d',
    )

    codes['v7.6.4-cuda92'] = cudnn_base.format(
        cudnn='cudnn-9.2-linux-x64-v7.6.4.38',
        cudnn_ver='v7.6.4',
        sha256sum='c79156531e641289b6a6952888b9637059ef30defd43c3cf82acf38d67f60a27',
    )

    codes['v7.6.4-cuda10'] = cudnn_base.format(
        cudnn='cudnn-10.0-linux-x64-v7.6.4.38',
        cudnn_ver='v7.6.4',
        sha256sum='417bb5daf51377037eb2f5c87649000ca1b9cec0acb16cfe07cb1d3e9a961dbf',
    )

    codes['v7.6.4-cuda101'] = cudnn_base.format(
        cudnn='cudnn-10.1-linux-x64-v7.6.4.38',
        cudnn_ver='v7.6.4',
        sha256sum='32091d115c0373027418620a09ebec3658a6bc467d011de7cdd0eb07d644b099',
    )

    codes['v7.6.5-cuda9'] = cudnn_base.format(
        cudnn='cudnn-9.0-linux-x64-v7.6.5.32',
        cudnn_ver='v7.6.5',
        sha256sum='bd0a4c0090d5b02feec3f195738968690cc2470b9bc6026e6fe8ff245cd261c8',
    )

    codes['v7.6.5-cuda92'] = cudnn_base.format(
        cudnn='cudnn-9.2-linux-x64-v7.6.5.32',
        cudnn_ver='v7.6.5',
        sha256sum='a2a2c7a8ba7b16d323b651766ee37dcfdbc2b50d920f73f8fde85005424960e4',
    )

    codes['v7.6.5-cuda10'] = cudnn_base.format(
        cudnn='cudnn-10.0-linux-x64-v7.6.5.32',
        cudnn_ver='v7.6.5',
        sha256sum='28355e395f0b2b93ac2c83b61360b35ba6cd0377e44e78be197b6b61b4b492ba',
    )

    codes['v7.6.5-cuda101'] = cudnn_base.format(
        cudnn='cudnn-10.1-linux-x64-v7.6.5.32',
        cudnn_ver='v7.6.5',
        sha256sum='7eaec8039a2c30ab0bc758d303588767693def6bf49b22485a2c00bf2e136cb3',
    )

    codes['v7.6.5-cuda102'] = cudnn_base.format(
        cudnn='cudnn-10.2-linux-x64-v7.6.5.32',
        cudnn_ver='v7.6.5',
        sha256sum='600267f2caaed2fd58eb214ba669d8ea35f396a7d19b94822e6b36f9f7088c20',
    )

    codes['v8.0.2-cuda101'] = cudnn_base.format(
        cudnn='cudnn-10.1-linux-x64-v8.0.2.39',
        cudnn_ver='v8.0.2',
        sha256sum='82148a68bd6bdaab93af5e05bb1842b8ccb3ab7de7bed41f609a7616c102213d',
    )

    codes['v8.0.2-cuda102'] = cudnn_base.format(
        cudnn='cudnn-10.2-linux-x64-v8.0.2.39',
        cudnn_ver='v8.0.2',
        sha256sum='c9cbe5c211360f3cfbc0fb104f0e9096b37e53f89392525679f049276b2f701f',
    )

    codes['v8.0.2-cuda11'] = cudnn_base.format(
        cudnn='cudnn-11.0-linux-x64-v8.0.2.39',
        cudnn_ver='v8.0.2',
        sha256sum='672f46288b8edd98f8d156a4f1ff518201ca6de0cff67915ceaa37f6d6d86345',
    )

    codes['v8.0.3-cuda101'] = cudnn_base.format(
        cudnn='cudnn-10.1-linux-x64-v8.0.3.33',
        cudnn_ver='v8.0.3',
        sha256sum='4752ac6aea4e4d2226061610d6843da6338ef75a93518aa9ce50d0f58df5fb07',
    )

    codes['v8.0.3-cuda102'] = cudnn_base.format(
        cudnn='cudnn-10.2-linux-x64-v8.0.3.33',
        cudnn_ver='v8.0.3',
        sha256sum='b3d487c621e24b5711983b89bb8ad34f0378bdbf8a1a4b86eefaa23b19956dcc',
    )

    codes['v8.0.3-cuda11'] = cudnn_base.format(
        cudnn='cudnn-11.0-linux-x64-v8.0.3.33',
        cudnn_ver='v8.0.3',
        sha256sum='8924bcc4f833734bdd0009050d110ad0c8419d3796010cf7bc515df654f6065a',
    )

    codes['v8.0.4-cuda101'] = cudnn_base.format(
        cudnn='cudnn-10.1-linux-x64-v8.0.4.30',
        cudnn_ver='v8.0.4',
        sha256sum='eb4b888e61715168f57a0a0a21c281ada6856b728e5112618ed15f8637487715',
    )

    codes['v8.0.4-cuda102'] = cudnn_base.format(
        cudnn='cudnn-10.2-linux-x64-v8.0.4.30',
        cudnn_ver='v8.0.4',
        sha256sum='c12c69eb16698eacac40aa46b9ce399d4cd86efb6ff0c105142f8a28fcfb980e',
    )

    codes['v8.0.4-cuda11'] = cudnn_base.format(
        cudnn='cudnn-11.0-linux-x64-v8.0.4.30',
        cudnn_ver='v8.0.4',
        sha256sum='38a81a28952e314e21577432b0bab68357ef9de7f6c8858f721f78df9ee60c35',
    )

    codes['v8.0.4-cuda111'] = cudnn_base.format(
        cudnn='cudnn-11.1-linux-x64-v8.0.4.30',
        cudnn_ver='v8.0.4',
        sha256sum='8f4c662343afce5998ce963500fe3bb167e9a508c1a1a949d821a4b80fa9beab',
    )

    codes['v8.0.5-cuda101'] = cudnn_base.format(
        cudnn='cudnn-10.1-linux-x64-v8.0.5.39',
        cudnn_ver='v8.0.5',
        sha256sum='90908495298896b33aa95063a3471f93c36627d7ac01c17dc36d75c65eea4a00',
    )

    codes['v8.0.5-cuda102'] = cudnn_base.format(
        cudnn='cudnn-10.2-linux-x64-v8.0.5.39',
        cudnn_ver='v8.0.5',
        sha256sum='21f84c05c67bf1ec859e77c38ccd5bf154964fa1c308f449959be4c356e382f3',
    )

    codes['v8.0.5-cuda11'] = cudnn_base.format(
        cudnn='cudnn-11.0-linux-x64-v8.0.5.39',
        cudnn_ver='v8.0.5',
        sha256sum='4e16ee7895deb4a8b1c194b812ba49586ef7d26902051401d3717511898a9b73',
    )

    codes['v8.0.5-cuda111'] = cudnn_base.format(
        cudnn='cudnn-11.1-linux-x64-v8.0.5.39',
        cudnn_ver='v8.0.5',
        sha256sum='1d046bfa79399dabcc6f6cb1507918754439442ea0ca9e0fbecdd446f9b00cce',
    )

    codes['v8.1.0-cuda102'] = cudnn_base.format(
        cudnn='cudnn-10.2-linux-x64-v8.1.0.77',
        cudnn_ver='v8.1.0',
        sha256sum='c5bc617d89198b0fbe485156446be15a08aee37f7aff41c797b120912f2b14b4',
    )

    codes['v8.1.0-cuda112'] = cudnn_base.format(
        cudnn='cudnn-11.2-linux-x64-v8.1.0.77',
        cudnn_ver='v8.1.0',
        sha256sum='dbe82faf071d91ba9bcf00480146ad33f462482dfee56caf4479c1b8dabe3ecb',
    )

    codes['v8.1.1-cuda102'] = cudnn_base.format(
        cudnn='cudnn-10.2-linux-x64-v8.1.1.33',
        cudnn_ver='v8.1.1',
        sha256sum='2a4a7b99a6e9bfa690eb19bb41e49553f2a7a491a5b3abfcae900e166c5b6ebd',
    )

    codes['v8.1.1-cuda112'] = cudnn_base.format(
        cudnn='cudnn-11.2-linux-x64-v8.1.1.33',
        cudnn_ver='v8.1.1',
        sha256sum='98a8784e92862f20018d20c281b30d4a0cd951f93694f6433ccf4ae9c502ba6a',
    )

    codes['v8.2.0-cuda102'] = cudnn_base.format(
        cudnn='cudnn-10.2-linux-x64-v8.2.0.53',
        cudnn_ver='v8.2.0',
        sha256sum='6ecbc98b3795e940ce0831ffb7cd2c0781830fdd6b1911f950bcaf6d569f807c',
    )

    codes['v8.2.0-cuda113'] = cudnn_base.format(
        cudnn='cudnn-11.3-linux-x64-v8.2.0.53',
        cudnn_ver='v8.2.0',
        sha256sum='7a195dc93a7cda2bdd4d9b73958d259c784be422cd941a9a625aab75309f19dc',
    )

    codes['v8.2.1-cuda102'] = cudnn_base.format(
        cudnn='cudnn-10.2-linux-x64-v8.2.1.32',
        cudnn_ver='v8.2.1',
        sha256sum='fd6321ff3bce4ce0cb3342e5bd38c96dcf3b073d44d0808962711c518b6d61e2',
    )

    codes['v8.2.1-cuda113'] = cudnn_base.format(
        cudnn='cudnn-11.3-linux-x64-v8.2.1.32',
        cudnn_ver='v8.2.1',
        sha256sum='39412acd9ef5dd27954b6b9f5df75bd381c5d7ceb7979af6c743a7f4521f9c77',
    )

    codes['v8.2.2-cuda102'] = cudnn_base.format(
        cudnn='cudnn-10.2-linux-x64-v8.2.2.26',
        cudnn_ver='v8.2.2',
        sha256sum='b4a2067774f509e65a1d8ba3bd86162b9e09de5946bb636887b4cc605dddeb6e',
    )

    codes['v8.2.2-cuda114'] = cudnn_base.format(
        cudnn='cudnn-11.4-linux-x64-v8.2.2.26',
        cudnn_ver='v8.2.2',
        sha256sum='fbc631ce19688e87d7d2420403b20db97885b17f718f0f51d7e9fc0905d86e07',
    )

    codes['v8.2.4-cuda102'] = cudnn_base.format(
        cudnn='cudnn-10.2-linux-x64-v8.2.4.15',
        cudnn_ver='v8.2.4',
        sha256sum='d23c94a3115a1c77116a6c127d9175fbf59f723364374f26a34699075f3222f1',
    )

    codes['v8.2.4-cuda114'] = cudnn_base.format(
        cudnn='cudnn-11.4-linux-x64-v8.2.4.15',
        cudnn_ver='v8.2.4',
        sha256sum='0e5d2df890b9967efa6619da421310d97323565a79f05a1a8cb9b7165baad0d7',
    )

    codes['v8.3.0-cuda102'] = cudnn_base.format(
        cudnn='cudnn-10.2-linux-x64-v8.3.0.98',
        cudnn_ver='v8.3.0',
        sha256sum='8d00144181808aee2b77d1864993227eed0e22ae580e6d6643972581ae509485',
    )

    codes['v8.3.0-cuda115'] = cudnn_base.format(
        cudnn='cudnn-11.5-linux-x64-v8.3.0.98',
        cudnn_ver='v8.3.0',
        sha256sum='44c6f5ad5cb12fb74fa0c9a1e7caaf4f04d755adf11a2e5b6c9e417da376b2b2',
    )

    cudnn83x_base = '''
curl -o {cudnn}.tar.xz https://developer.download.nvidia.com/compute/redist/cudnn/{cudnn_ver}/local_installers/{cuda_ver}/{cudnn}.tar.xz &&
echo "{sha256sum}  {cudnn}.tar.xz" | sha256sum -cw --quiet - &&
tar -xJf {cudnn}.tar.xz -C {{path}} &&
rm {cudnn}.tar.xz
'''

    codes['v8.3.1-cuda102'] = cudnn83x_base.format(
        cudnn='cudnn-linux-x86_64-8.3.1.22_cuda10.2-archive',
        cuda_ver='10.2',
        cudnn_ver='v8.3.1',
        sha256sum='5982bb96c2a720268fa44b908feb5258d060ad47f1f6e6030e760d13195ea964',
    )

    codes['v8.3.1-cuda115'] = cudnn83x_base.format(
        cudnn='cudnn-linux-x86_64-8.3.1.22_cuda11.5-archive',
        cuda_ver='11.5',
        cudnn_ver='v8.3.1',
        sha256sum='f5ff3c69b6a8a9454289b42eca1dd41c3527f70fcf49428eb80502bcf6b02f6e',
    )

    codes['v8.3.2-cuda102'] = cudnn83x_base.format(
        cudnn='cudnn-linux-x86_64-8.3.2.44_cuda10.2-archive',
        cuda_ver='10.2',
        cudnn_ver='v8.3.2',
        sha256sum='d6f56ef9ca8cf8f91eb73210ba6c3dca49ba4446c1661bfafe55c1ec40b669ac',
    )

    codes['v8.3.2-cuda115'] = cudnn83x_base.format(
        cudnn='cudnn-linux-x86_64-8.3.2.44_cuda11.5-archive',
        cuda_ver='11.5',
        cudnn_ver='v8.3.2',
        sha256sum='5500953c08c5e5d1dddcfda234f9efbddcdbe43a53b26dc0a82c723fa170c457',
    )

    codes['v8.3.3-cuda102'] = cudnn83x_base.format(
        cudnn='cudnn-linux-x86_64-8.3.3.40_cuda10.2-archive',
        cuda_ver='10.2',
        cudnn_ver='v8.3.3',
        sha256sum='d8554f2b32e6295d5fc8f3ac25e68f94058b018c801dab9c143e36812f8926ab',
    )

    codes['v8.3.3-cuda115'] = cudnn83x_base.format(
        cudnn='cudnn-linux-x86_64-8.3.3.40_cuda11.5-archive',
        cuda_ver='11.5',
        cudnn_ver='v8.3.3',
        sha256sum='eabe96c75cf03ea4f5379894d914f1f8ae14ceab121989e84b0836d927fb7731',
    )

    codes['v8.4.0-cuda102'] = cudnn83x_base.format(
        cudnn='cudnn-linux-x86_64-8.4.0.27_cuda10.2-archive',
        cuda_ver='10.2',
        cudnn_ver='v8.4.0',
        sha256sum='14c5e3ca4258271996d1fd959c42d17c582ce4d9aff451f84524469e784fd154',
    )

    codes['v8.4.0-cuda116'] = cudnn83x_base.format(
        cudnn='cudnn-linux-x86_64-8.4.0.27_cuda11.6-archive',
        cuda_ver='11.6',
        cudnn_ver='v8.4.0',
        sha256sum='d19bdafd9800c79d29e6f6fffa9f9e2c10d1132d6c2ff10b1593e057e74dd050',
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
    codes['v74-cuda9'] = codes['v7.4.2-cuda9']
    codes['v74-cuda92'] = codes['v7.4.2-cuda92']
    codes['v74-cuda10'] = codes['v7.4.2-cuda10']
    codes['v75-cuda9'] = codes['v7.5.1-cuda9']
    codes['v75-cuda92'] = codes['v7.5.1-cuda92']
    codes['v75-cuda10'] = codes['v7.5.1-cuda10']
    codes['v75-cuda101'] = codes['v7.5.1-cuda101']
    codes['v76-cuda9'] = codes['v7.6.5-cuda9']
    codes['v76-cuda92'] = codes['v7.6.5-cuda92']
    codes['v76-cuda10'] = codes['v7.6.5-cuda10']
    codes['v76-cuda101'] = codes['v7.6.5-cuda101']
    codes['v76-cuda102'] = codes['v7.6.5-cuda102']
    codes['v8-cuda101'] = codes['v8.0.5-cuda101']
    codes['v8-cuda102'] = codes['v8.0.5-cuda102']
    codes['v8-cuda11'] = codes['v8.0.5-cuda11']
    codes['v8-cuda111'] = codes['v8.0.5-cuda111']
    codes['v81-cuda102'] = codes['v8.1.1-cuda102']
    codes['v81-cuda112'] = codes['v8.1.1-cuda112']
    codes['v82-cuda102'] = codes['v8.2.4-cuda102']
    codes['v82-cuda114'] = codes['v8.2.4-cuda114']
    codes['v83-cuda102'] = codes['v8.3.3-cuda102']
    codes['v83-cuda115'] = codes['v8.3.3-cuda115']
    codes['v84-cuda102'] = codes['v8.4.0-cuda102']
    codes['v84-cuda116'] = codes['v8.4.0-cuda116']

    LIBDIR = 'lib64'

elif sys.platform == 'darwin':
    cudnn2_base = '''
curl -s -o {cudnn}.tgz https://developer.download.nvidia.com/compute/redist/cudnn/{cudnn_ver}/{cudnn}.tgz &&
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
curl -o {cudnn}.tgz https://developer.download.nvidia.com/compute/redist/cudnn/{cudnn_ver}/{cudnn}.tgz &&
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

    codes['v7.4.1-cuda10'] = cudnn_base.format(
        cudnn='cudnn-10.0-osx-x64-v7.4.1.5',
        cudnn_ver='v7.4.1',
        sha256sum='82cdd28ea38724e0be4cac8dd67cec5ec2936a662a992a8740682e837972ec2d',
    )

    codes['v7.4.2-cuda10'] = cudnn_base.format(
        cudnn='cudnn-10.0-osx-x64-v7.4.2.24',
        cudnn_ver='v7.4.2',
        sha256sum='4b1c53af7e50ff046ed53d4ba9129900da9bef0008b93cefad2e4195f1ee8ebb',
    )

    codes['v7.5.0-cuda10'] = cudnn_base.format(
        cudnn='cudnn-10.0-osx-x64-v7.5.0.56',
        cudnn_ver='v7.5.0',
        sha256sum='5f72e5776eef6a4c5fc5c677404511faabc929864bd0886b884560e5bcdc8f54',
    )

    codes['v7.5.0-cuda101'] = cudnn_base.format(
        cudnn='cudnn-10.1-osx-x64-v7.5.0.56',
        cudnn_ver='v7.5.0',
        sha256sum='58f7e5b51e3229cb294c1c134eef3ca1a0be1b3d12c3277d24d1912051f54840',
    )

    codes['v7.5.1-cuda10'] = cudnn_base.format(
        cudnn='cudnn-10.0-osx-x64-v7.5.1.10',
        cudnn_ver='v7.5.1',
        sha256sum='152aaaba608f1cb55bc84fa51b26bb7515ae9b791d2a847a542c48b57d52f540',
    )

    codes['v7.5.1-cuda101'] = cudnn_base.format(
        cudnn='cudnn-10.1-osx-x64-v7.5.1.10',
        cudnn_ver='v7.5.1',
        sha256sum='2035071b11ffd7da8473f47a63ffd6610cb73108e072ff44b708ec833c218b45',
    )

    codes['v7.6.0-cuda10'] = cudnn_base.format(
        cudnn='cudnn-10.0-osx-x64-v7.6.0.64',
        cudnn_ver='v7.6.0',
        sha256sum='30ea0ef517d1f8a2be29a13bdefa9abd71d952d0cddfb8b2c3fba36a48c46f0f',
    )

    codes['v7.6.0-cuda101'] = cudnn_base.format(
        cudnn='cudnn-10.1-osx-x64-v7.6.0.64',
        cudnn_ver='v7.6.0',
        sha256sum='8d1417e8f528cfda4e859cf41ecb843bad0a8920935ce0dadde5f7327d8427fb',
    )

    codes['v7.6.1-cuda10'] = cudnn_base.format(
        cudnn='cudnn-10.0-osx-x64-v7.6.1.34',
        cudnn_ver='v7.6.1',
        sha256sum='a8e18c9f0b9641e0d35369ce8be36ab22aa45b979275b55fce86a09918922bd5',
    )

    codes['v7.6.1-cuda101'] = cudnn_base.format(
        cudnn='cudnn-10.1-osx-x64-v7.6.1.34',
        cudnn_ver='v7.6.1',
        sha256sum='22f7783cf5a1c11f53d040e16fdf5210f9a4e7db0f610e9b4674651801a667bc',
    )

    codes['v7.6.2-cuda10'] = cudnn_base.format(
        cudnn='cudnn-10.0-osx-x64-v7.6.2.24',
        cudnn_ver='v7.6.2',
        sha256sum='412d3790c496e518bbcde56f6125b2cecc0c93ad87b91431f87b753b1d2f01df',
    )

    codes['v7.6.2-cuda101'] = cudnn_base.format(
        cudnn='cudnn-10.1-osx-x64-v7.6.2.24',
        cudnn_ver='v7.6.2',
        sha256sum='84c3298bad04b2944350001a9302b200846ccf3c41e01d7269027b52352608b9',
    )

    codes['v7.6.3-cuda10'] = cudnn_base.format(
        cudnn='cudnn-10.0-osx-x64-v7.6.3.30',
        cudnn_ver='v7.6.3',
        sha256sum='2b93e5c07345b08dec4b7bb5ed0c41b78ab39f413ae93edc881fbbfcba00976c',
    )

    codes['v7.6.3-cuda101'] = cudnn_base.format(
        cudnn='cudnn-10.1-osx-x64-v7.6.3.30',
        cudnn_ver='v7.6.3',
        sha256sum='081703baa7c117d93ab0c753e4029c5b3c6f9517cde2ce4f8bebd324a67900f0',
    )

    codes['v7.6.4-cuda10'] = cudnn_base.format(
        cudnn='cudnn-10.0-osx-x64-v7.6.4.38',
        cudnn_ver='v7.6.4',
        sha256sum='af01ab841caec25087776a6b8fc7782883da12e590e24825ad1031f9ae0ed4b1',
    )

    codes['v7.6.4-cuda101'] = cudnn_base.format(
        cudnn='cudnn-10.1-osx-x64-v7.6.4.38',
        cudnn_ver='v7.6.4',
        sha256sum='bfced062c3689ced2c1fb49c7d5052e6bc3da6974c1eb707e4dcf8cd209d4236',
    )

    codes['v7.6.5-cuda10'] = cudnn_base.format(
        cudnn='cudnn-10.0-osx-x64-v7.6.5.32',
        cudnn_ver='v7.6.5',
        sha256sum='6fa0b819374da49102e285ecf7fcb8879df4d0b3cc430cc8b781cdeb41009b47',
    )

    codes['v7.6.5-cuda101'] = cudnn_base.format(
        cudnn='cudnn-10.1-osx-x64-v7.6.5.32',
        cudnn_ver='v7.6.5',
        sha256sum='8ecce28a5ed388a2b9b2d239e08d7c550f53b79288e6d9e5eb4c152bfc711aff',
    )

    codes['v7-cuda9'] = codes['v7.0.3-cuda9']
    codes['v71-cuda92'] = codes['v7.1.4-cuda92']
    codes['v72-cuda92'] = codes['v7.2.1-cuda92']
    codes['v73-cuda10'] = codes['v7.3.0-cuda10']
    codes['v74-cuda10'] = codes['v7.4.2-cuda10']
    codes['v75-cuda10'] = codes['v7.5.1-cuda10']
    codes['v75-cuda101'] = codes['v7.5.1-cuda101']
    codes['v76-cuda10'] = codes['v7.6.5-cuda10']
    codes['v76-cuda101'] = codes['v7.6.5-cuda101']

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
