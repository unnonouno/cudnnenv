try:
    import io
    StringIO = io.StringIO
except ImportError:
    import StringIO
    StringIO = StringIO.StringIO
import shutil
import sys
import tempfile

import cudnnenv


class TestBase(object):

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
