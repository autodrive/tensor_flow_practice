import unittest

import fizzbuzz as fb


class TestFizzBuzz(unittest.TestCase):
    def test_binary_encode(self):
        n_table = [0, 1, 2, 2, 3, ]
        for i in range(0, 5):

            for n in range(i + 2, i + 4):
                result = fb.binary_encode(i, n)
                # make following line True to generate docstring
                if False:
                    # write a doc string
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

    def test_fizz_buzz_encode(self):
        result_15 = fb.fizz_buzz_encode(15)
        print('''>>> fizz_buzz_encode(%d)
%r''' % (15, result_15))
if __name__ == '__main__':
    # help(unittest.TestCase)
    unittest.main()
