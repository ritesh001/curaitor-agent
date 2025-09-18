# import unittest

# class TestA(unittest.TestCase):
#     def test_funct(self):                       
#         self.assertEqual(1,1, msg="message")    


def add(a, b):
    return a + b
               

def test_add():
    assert add(1, 2) == 3
    assert add(-1, 5) == 4