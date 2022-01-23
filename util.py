# From LeastSquares.py, https://ese224.seas.upenn.edu/wp-content/uploads/2021/01/oop_python.zip
import numpy as np
################################################################################
#
# A function to print a matrix (or vector). It formats the numbers and prints
# the matrix in a way that makes it easy to read. It is not important.
#
# The arguments we pass to the function are:
#
#     A = The matrix to be printed
#     nr_decimals = the number of decimals to be printed. It is an optional
#                   argument. Set to 2 by default

def print_matrix(A, nr_decimals=2):
    # Determine the number of digits in the largest number in the matrix and use
    # it to specify the number format

    nr_digits = np.maximum(np.floor(np.log10(np.amax(np.abs(A)))), 0) + 1
    nr_digits = nr_digits + nr_decimals + 3
    nr_digits = "{0:1.0f}".format(nr_digits)
    number_format = "{0: " + nr_digits + "." + str(nr_decimals) + "f}"

    # Determine matrix size
    n = len(A)
    m = len(A[0])

    # Sweep through rows
    for l in range(m):
        value = " "

        # Sweep through columns
        for k in range(n):
            # ccncatenate entries to create row printout
            value = value + " " + number_format.format(A[k, l])

        # Print row
        print(value)