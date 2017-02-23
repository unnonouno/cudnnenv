import unittest

from test import base


class TestUninstall(base.TestBase, unittest.TestCase):

    def test_clean_environment(self):
        with self.assertRaises(SystemExit) as cont:
            self.call_main('uninstall', 'v2')

        self.assertEqual(cont.exception.code, 2)
