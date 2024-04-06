import unittest
from bubble import bubbleSort


class TestBubbleSort(unittest.TestCase):

    def test_types(self):
        self.assertRaises(TypeError, bubbleSort, False)
        self.assertRaises(TypeError, bubbleSort, 2j)
        self.assertRaises(TypeError, bubbleSort, 21)
        self.assertRaises(TypeError, bubbleSort, [4.3, 5.2])
        self.assertRaises(TypeError, bubbleSort, [1, 2, 3, 4, 5], 6)
        self.assertRaises(TypeError, bubbleSort, 'eight')
        self.assertRaises(TypeError, bubbleSort, ['Vasya', 1, 2, 3, 3j], 4, 35, 12)
        self.assertRaises(TypeError, bubbleSort, [[]])

    def test_input_limit(self):
        input_limit = 10
        arr = [1] * (input_limit + 1)
        with self.assertRaises(Exception):
            bubbleSort(arr)

    def test_values_Min_and_Max(self):
        with self.assertRaises(ValueError) as i:
            bubbleSort([1000001, -1001])
        self.assertEqual('value amount exceeds max', i.exception.args[0])


    def test_sorted_array(self):
        not_sorted_list = [2, -24, 13839, 273, -383, 82, 3, 2, 5]
        expected_sorted_list = [-383, -24, 2, 2, 3, 5, 82, 273, 13839]
        self.assertEqual(bubbleSort(not_sorted_list), expected_sorted_list )


if __name__ == "__main__":
    unittest.main()
