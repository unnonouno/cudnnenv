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

import cudnnenv


class TestSafeTempDir(unittest.TestCase):

    def test_safe_temp_dir(self):
        with cudnnenv.safe_temp_dir() as path:
            self.assertTrue(os.path.exists(path))
        self.assertFalse(os.path.exists(path))

    def test_safe_temp_dir_error(self):
        try:
            with cudnnenv.safe_temp_dir() as path:
                raise Exception
        except Exception:
            pass
        self.assertFalse(os.path.exists(path))


class TestSafeDir(unittest.TestCase):

    def setUp(self):
        self.path = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.path, ignore_errors=True)

    def test_safe_dir(self):
        path = os.path.join(self.path, 'd')
        with cudnnenv.safe_dir(path) as p:
            self.assertTrue(os.path.exists(p))
        self.assertTrue(os.path.exists(path))

    def test_safe_dir_error(self):
        path = os.path.join(self.path, 'd')
        try:
            with cudnnenv.safe_dir(path) as p:
                raise Exception
        except Exception:
            pass
        self.assertFalse(os.path.exists(p))
        self.assertFalse(os.path.exists(path))


class TestYesNo(unittest.TestCase):

    def tearDown(self):
        sys.stdin = sys.__stdin__

    def test_yes(self):
        sys.stdin = StringIO('y\n')
        self.assertTrue(cudnnenv.yes_no_query('q'))

    def test_no(self):
        sys.stdin = StringIO('n\n')
        self.assertFalse(cudnnenv.yes_no_query('q'))

    def test_invalid(self):
        sys.stdin = StringIO('a\nb\nc\nd\ny\nn\n')
        self.assertTrue(cudnnenv.yes_no_query('q'))
