import unittest

from test import base


class TestInstall(base.TestBase, unittest.TestCase):

    def test_unknown(self):
        with self.assertRaises(SystemExit) as cont:
            self.call_main('install', 'unknown')

        self.assertEqual(cont.exception.code, 2)
