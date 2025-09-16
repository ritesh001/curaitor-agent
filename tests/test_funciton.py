import unittest

class TestA(unittest.TestCase):
    def test_funct(self):                       
        self.assertEqual(1,1, msg="message")                   