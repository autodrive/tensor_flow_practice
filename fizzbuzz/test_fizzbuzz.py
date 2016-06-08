import unittest

import fizzbuzz as fb


class TestFizzBuzz(unittest.TestCase):
    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_binary_encode(self):
        n_table = [0, 1, 2, 2, 3, ]
        for i in range(0, 5):

            for n in range(i + 2, i + 4):
                result = fb.binary_encode(i, n)
                print('''>>> binary_encode(%d, %d)
%s''' % (i, n, result))
                self.assertEqual(len(result), n)

                result_to_int = sum([(2 ** p) * d for p, d in enumerate(result)])

                if result_to_int != i:
                    print ()

                try:
                    self.assertEqual(result_to_int, i)
                except TypeError as e:
                    print e
                    raise e


if __name__ == '__main__':
    # help(unittest.TestCase)
    unittest.main()
