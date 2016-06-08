import unittest

import fizzbuzz as fb


class TestFizzBuzz(unittest.TestCase):
    def test_binary_encode(self):
        n_table = [0, 1, 2, 2, 3, ]
        for i in range(0, 5):
            # n starts from a positive integer to avoid result_to_int conversion
            for n in range(i + 2, i + 4):
                result = fb.binary_encode(i, n)
                # make following line True to generate docstring
                if False:
                    doctest = generate_docstring('binary_encode', (i, n), result)
                    # write a doctest string
                    print(doctest)
                self.assertEqual(len(result), n)

                # try to reconstruct i from result
                # multiply (power of 2 at each position) and member of result at the position
                result_to_int = sum([(2 ** p) * d for p, d in enumerate(result)])

                self.assertEqual(result_to_int, i)

    def test_fizz_buzz_encode(self):
        for i, expected_list in ((0, [0, 0, 0, 1]),
                                 (3, [0, 1, 0, 0]),
                                 (5, [0, 0, 1, 0]),
                                 (15, [0, 0, 0, 1])):
            result_15 = fb.fizz_buzz_encode(i)
            self.assertSequenceEqual(result_15.tolist(), expected_list)

            if False:
                docstring = generate_docstring('fizz_buzz_encode', i, result_15)
                print(docstring)


def generate_docstring(function_under_test_name, arguments, result):
    # if the arguments is just a single digit, make it a list
    if len not in dir(arguments):
        arguments = [arguments]
    # if the arguments is just a single string, make it a list
    elif isinstance(arguments, str):
        arguments = [arguments]

    # docstring is in the form of
    # >>> math.sin(0)
    # 0.0
    # 'arguments' is assumed to be a list. making it a string will fit into () for function call
    # but with '[' and ']'
    docstring = '''>>> %s(%s)\n%r''' % (
        function_under_test_name, str(arguments)[1:-1], result)
    return docstring


if __name__ == '__main__':
    # help(unittest.TestCase)
    unittest.main()
