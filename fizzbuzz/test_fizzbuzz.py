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
                    doctest = generate_docstring('binary_encode', (i, n), result)
                    # write a doctest string
                    print(doctest)
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

        function_under_test_name = 'fizz_buzz_encode'
        arguments = 15

        docstring = generate_docstring(function_under_test_name, arguments, result_15)
        print(docstring)


def generate_docstring(function_under_test_name, arguments, result):
    # if the arguments is just a single digit, make it a list
    if len not in dir(arguments):
        arguments = [arguments]
    # if the arguments is just a single string, make it a list
    elif isinstance(arguments, str):
        arguments = [arguments]


    docstring = '''>>> %s(%s)\n%r''' % (
        function_under_test_name, str(arguments)[1:-1], result)
    return docstring


if __name__ == '__main__':
    # help(unittest.TestCase)
    unittest.main()
