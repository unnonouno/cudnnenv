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
        cudnnenv.cudnn_home = self.original_cudnn_home

    def call_main(self, *args):
        return cudnnenv.main(args)

    def get_stdout(self):
        return self.stdout.getvalue()

    def clear_stdout(self):
        self.stdout = StringIO()
        sys.stdout = self.stdout

    def test_install_unknown(self):
        with self.assertRaises(SystemExit) as cont:
            self.call_main('install', 'unknown')

        self.assertEqual(cont.exception.code, 2)

    def test_install_file(self):
        self.call_main('install-file', self.empty_tgz_path, 'v0')

        self.assertTrue(os.path.exists(os.path.join(self.path, 'active')))
        self.assertTrue(os.path.exists(
            os.path.join(self.path, 'versions', 'v0')))
        active = os.readlink(os.path.join(self.path, 'active'))
        self.assertEqual(active, 'versions/v0')

        self.clear_stdout()
        self.call_main('version')
        self.assertEqual(self.get_stdout(), 'v0\n')

    def test_install_exists(self):
        self.call_main('install-file', self.empty_tgz_path, 'v0')
        with self.assertRaises(SystemExit) as cont:
            self.call_main('install-file', self.empty_tgz_path, 'v0')
        self.assertEqual(cont.exception.code, 3)

    def test_clean_environment(self):
        self.call_main('versions')
        self.assertEqual(self.get_stdout(), '''Available versions:
  v2
  v3
  v4
  v5
  v5-cuda8
  v51
  v51-cuda8

Installed versions:
''')

        self.clear_stdout()
        self.call_main('version')
        self.assertEqual(self.get_stdout(), '(none)\n')

        self.call_main('deactivate')

        with self.assertRaises(SystemExit) as cont:
            self.call_main('uninstall', 'v2')
        self.assertEqual(cont.exception.code, 2)

    def test_uninstall(self):
        self.call_main('install-file', self.empty_tgz_path, 'v0')
        with mock.patch('__builtin__.raw_input', return_value='y'):
            self.call_main('uninstall', 'v0')
