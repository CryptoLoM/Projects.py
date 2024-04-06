class InputSizeLimitExceeded(Exception):
    pass



def bubbleSort(array):
    n = len(array)

    # Sorting algorithm
    for i in range(n - 1):
        for j in range(0, n - i - 1):
            if array[j] > array[j + 1]:
                array[j], array[j + 1] = array[j + 1], array[j]  # Swap elements

    # Test for input size limit
    input_limit = 10
    if len(array) > input_limit:
        raise InputSizeLimitExceeded("Input size exceeds limit")

    # Test for types
    for el in array:
        if not isinstance(el, int):
            raise TypeError('Unexpected types')

    # Test for maximal el
    for el in array:
        if not el <= 1000000:
            raise ValueError('value amount exceeds max')

    # Test for minimal el
    for el in array:
        if not el >= -1000:
            raise ValueError('value amount exceeds min')

    return array


arr = [24992, 43786, 94378, 91694, 9, 12, 203, -376]

bubbleSort(arr)


print("Відсортований список:")
for element in arr:
    print(element, end=" ")
