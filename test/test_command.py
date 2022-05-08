from __future__ import unicode_literals

try:
    import io
    StringIO = io.StringIO
except ImportError:
    import StringIO
    StringIO = StringIO.StringIO
import os
import shutil
import sys
import tempfile
import unittest

import mock

import cudnnenv


if 'linux' in sys.platform:
    _available_versions = '''  v2
  v3
  v4
  v5
  v5-cuda8
  v51
  v51-cuda8
  v6
  v6-cuda8
  v7-cuda8
  v7-cuda9
  v7-cuda91
  v7.0.1-cuda8
  v7.0.1-cuda9
  v7.0.2-cuda8
  v7.0.2-cuda9
  v7.0.3-cuda8
  v7.0.3-cuda9
  v7.0.4-cuda8
  v7.0.4-cuda9
  v7.0.5-cuda8
  v7.0.5-cuda9
  v7.0.5-cuda91
  v7.1.1-cuda8
  v7.1.1-cuda9
  v7.1.1-cuda91
  v7.1.2-cuda8
  v7.1.2-cuda9
  v7.1.2-cuda91
  v7.1.3-cuda8
  v7.1.3-cuda9
  v7.1.3-cuda91
  v7.1.4-cuda9
  v7.1.4-cuda92
  v7.2.1-cuda8
  v7.2.1-cuda9
  v7.2.1-cuda92
  v7.3.0-cuda10
  v7.3.0-cuda9
  v7.4.1-cuda10
  v7.4.1-cuda9
  v7.4.1-cuda92
  v7.4.2-cuda10
  v7.4.2-cuda9
  v7.4.2-cuda92
  v7.5.0-cuda10
  v7.5.0-cuda101
  v7.5.0-cuda9
  v7.5.0-cuda92
  v7.5.1-cuda10
  v7.5.1-cuda101
  v7.5.1-cuda9
  v7.5.1-cuda92
  v7.6.0-cuda10
  v7.6.0-cuda101
  v7.6.0-cuda9
  v7.6.0-cuda92
  v7.6.1-cuda10
  v7.6.1-cuda101
  v7.6.1-cuda9
  v7.6.1-cuda92
  v7.6.2-cuda10
  v7.6.2-cuda101
  v7.6.2-cuda9
  v7.6.2-cuda92
  v7.6.3-cuda10
  v7.6.3-cuda101
  v7.6.3-cuda9
  v7.6.3-cuda92
  v7.6.4-cuda10
  v7.6.4-cuda101
  v7.6.4-cuda9
  v7.6.4-cuda92
  v7.6.5-cuda10
  v7.6.5-cuda101
  v7.6.5-cuda102
  v7.6.5-cuda9
  v7.6.5-cuda92
  v71-cuda8
  v71-cuda9
  v71-cuda91
  v71-cuda92
  v72-cuda8
  v72-cuda9
  v72-cuda92
  v73-cuda10
  v73-cuda9
  v74-cuda10
  v74-cuda9
  v74-cuda92
  v75-cuda10
  v75-cuda101
  v75-cuda9
  v75-cuda92
  v76-cuda10
  v76-cuda101
  v76-cuda102
  v76-cuda9
  v76-cuda92
  v8-cuda101
  v8-cuda102
  v8-cuda11
  v8-cuda111
  v8.0.2-cuda101
  v8.0.2-cuda102
  v8.0.2-cuda11
  v8.0.3-cuda101
  v8.0.3-cuda102
  v8.0.3-cuda11
  v8.0.4-cuda101
  v8.0.4-cuda102
  v8.0.4-cuda11
  v8.0.4-cuda111
  v8.0.5-cuda101
  v8.0.5-cuda102
  v8.0.5-cuda11
  v8.0.5-cuda111
  v8.1.0-cuda102
  v8.1.0-cuda112
  v8.1.1-cuda102
  v8.1.1-cuda112
  v8.2.0-cuda102
  v8.2.0-cuda113
  v8.2.1-cuda102
  v8.2.1-cuda113
  v8.2.2-cuda102
  v8.2.2-cuda114
  v8.2.4-cuda102
  v8.2.4-cuda114
  v8.3.0-cuda102
  v8.3.0-cuda115
  v8.3.1-cuda102
  v8.3.1-cuda115
  v8.3.2-cuda102
  v8.3.2-cuda115
  v8.3.3-cuda102
  v8.3.3-cuda115
  v8.4.0-cuda102
  v8.4.0-cuda116
  v81-cuda102
  v81-cuda112
  v82-cuda102
  v82-cuda114
  v83-cuda102
  v83-cuda115
  v84-cuda102
  v84-cuda116
'''
elif sys.platform == 'darwin':
    _available_versions = '''  v2
  v3
  v4
  v5
  v5-cuda8
  v51
  v51-cuda8
  v6
  v6-cuda8
  v7-cuda9
  v7.0.1-cuda9
  v7.0.2-cuda9
  v7.0.3-cuda9
  v7.1.4-cuda92
  v7.2.1-cuda92
  v7.3.0-cuda10
  v7.4.1-cuda10
  v7.4.2-cuda10
  v7.5.0-cuda10
  v7.5.0-cuda101
  v7.5.1-cuda10
  v7.5.1-cuda101
  v7.6.0-cuda10
  v7.6.0-cuda101
  v7.6.1-cuda10
  v7.6.1-cuda101
  v7.6.2-cuda10
  v7.6.2-cuda101
  v7.6.3-cuda10
  v7.6.3-cuda101
  v7.6.4-cuda10
  v7.6.4-cuda101
  v7.6.5-cuda10
  v7.6.5-cuda101
  v71-cuda92
  v72-cuda92
  v73-cuda10
  v74-cuda10
  v75-cuda10
  v75-cuda101
  v76-cuda10
  v76-cuda101
'''


class TestCommand(unittest.TestCase):

    empty_tgz_path = os.path.join(
        os.path.dirname(__file__), 'files', 'cudnn.empty.tar.gz')

    def setUp(self):
        self.path = tempfile.mkdtemp()
        self.original_cudnn_home = cudnnenv.cudnn_home
        cudnnenv.cudnn_home = self.path
        self.stdout = StringIO()
        sys.stdout = self.stdout

    def tearDown(self):
        shutil.rmtree(self.path, ignore_errors=True)
        sys.stdout = sys.__stdout__
        sys.stdin = sys.__stdin__
        cudnnenv.cudnn_home = self.original_cudnn_home

    def call_main(self, *args):
        return cudnnenv.main(args)

    def get_stdout(self):
        return self.stdout.getvalue()

    def set_stdin(self, s):
        sys.stdin = StringIO(s)

    def clear_stdout(self):
        self.stdout = StringIO()
        sys.stdout = self.stdout

    def test_version(self):
        with self.assertRaises(SystemExit):
            self.call_main('--version')

    def test_no_subcommand(self):
        with self.assertRaises(SystemExit) as cont:
            self.call_main()

        self.assertEqual(cont.exception.code, 2)

    def test_install_unknown(self):
        with self.assertRaises(SystemExit) as cont:
            self.call_main('install', 'unknown')

        self.assertEqual(cont.exception.code, 2)

    def test_install_interrupt(self):
        def interrupt(cmd, **kwargs):
            raise KeyboardInterrupt

        try:
            with mock.patch('subprocess.check_call', new=interrupt):
                self.call_main('install', 'v2')
        except BaseException:
            pass

        # Check if no directory is created
        self.assertFalse(os.path.exists(
            os.path.join(self.path, 'versions', 'v2')))

    def test_install_file_and_uninstall(self):
        self.call_main('install-file', self.empty_tgz_path, 'v0')

        self.assertTrue(os.path.exists(os.path.join(self.path, 'active')))
        self.assertTrue(os.path.exists(
            os.path.join(self.path, 'versions', 'v0')))
        active = os.readlink(os.path.join(self.path, 'active'))
        self.assertEqual(active, 'versions/v0')

        self.clear_stdout()
        self.call_main('version')
        self.assertEqual(self.get_stdout(), 'v0\n')

        self.clear_stdout()
        self.call_main('versions')
        self.assertEqual(self.get_stdout(), '''Available versions:
{}
Installed versions:
* v0
'''.format(_available_versions))

        self.set_stdin('y\n')
        self.call_main('uninstall', 'v0')
        self.assertFalse(os.path.exists(
            os.path.join(self.path, 'versions', 'v0')))

        self.clear_stdout()
        self.call_main('versions')
        self.assertEqual(self.get_stdout(), '''Available versions:
{}
Installed versions:
'''.format(_available_versions))

    def test_install_exists(self):
        self.call_main('install-file', self.empty_tgz_path, 'v0')
        with self.assertRaises(SystemExit) as cont:
            self.call_main('install-file', self.empty_tgz_path, 'v0')
        self.assertEqual(cont.exception.code, 3)

    def test_activate_deactivate(self):
        self.call_main('install-file', self.empty_tgz_path, 'v0')
        active = os.readlink(os.path.join(self.path, 'active'))
        self.assertEqual(active, 'versions/v0')

        self.call_main('install-file', self.empty_tgz_path, 'v1')
        active = os.readlink(os.path.join(self.path, 'active'))
        self.assertEqual(active, 'versions/v1')

        self.call_main('activate', 'v0')
        active = os.readlink(os.path.join(self.path, 'active'))
        self.assertEqual(active, 'versions/v0')

        self.call_main('deactivate')
        self.assertFalse(os.path.exists(os.path.join(self.path, 'active')))

        self.clear_stdout()
        self.call_main('versions')
        self.assertEqual(self.get_stdout(), '''Available versions:
{}
Installed versions:
  v0
  v1
'''.format(_available_versions))

    def test_clean_environment(self):
        self.call_main('versions')
        self.assertEqual(self.get_stdout(), '''Available versions:
{}
Installed versions:
'''.format(_available_versions))

        self.clear_stdout()
        self.call_main('version')
        self.assertEqual(self.get_stdout(), '(none)\n')

        self.call_main('deactivate')

        with self.assertRaises(SystemExit) as cont:
            self.call_main('uninstall', 'v2')
        self.assertEqual(cont.exception.code, 2)

    def test_unknown_subcommand(self):
        with self.assertRaises(SystemExit) as cont:
            self.call_main('unknown')
        self.assertEqual(cont.exception.code, 2)
