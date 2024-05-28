def matrix_addition(matrix1, matrix2):
    if len(matrix1) != len(matrix2) or len(matrix1[0]) != len(matrix2[0]):
        return "Matrices must have the same dimensions for addition."
    result = []
    for i in range(len(matrix1)):
        row = []
        for j in range(len(matrix1[0])):
            row.append(matrix1[i][j] + matrix2[i][j])
        result.append(row)
    return result

def matrix_subtraction(matrix1, matrix2):
    if len(matrix1) != len(matrix2) or len(matrix1[0]) != len(matrix2[0]):
        return "Matrices must have the same dimensions for subtraction."
    result = []
    for i in range(len(matrix1)):
        row = []
        for j in range(len(matrix1[0])):
            row.append(matrix1[i][j] - matrix2[i][j])
        result.append(row)
    return result

def scalar_multiplication(matrix, scalar):
    result = []
    for i in range(len(matrix)):
        row = []
        for j in range(len(matrix[0])):
            row.append(matrix[i][j] * scalar)
        result.append(row)
    return result

def matrix_multiplication(matrix1, matrix2):
    if len(matrix1[0]) != len(matrix2):
        return "Number of columns in the first matrix must be equal to the number of rows in the second matrix for multiplication."
    result = []
    for i in range(len(matrix1)):
        row = []
        for j in range(len(matrix2[0])):
            sum = 0
            for k in range(len(matrix2)):
                sum += matrix1[i][k] * matrix2[k][j]
            row.append(sum)
        result.append(row)
    return result

def matrix_transpose(matrix):
    result = []
    for j in range(len(matrix[0])):
        row = []
        for i in range(len(matrix)):
            row.append(matrix[i][j])
        result.append(row)
    return result

matrix_a = [[1, 2, 3], [4, 5, 6]]
matrix_b = [[7, 8, 9], [10, 11, 12]]
print("Matrix A:")
for row in matrix_a:
    print(row)
print("\nMatrix B:")
for row in matrix_b:
    print(row)

matrix_c = [[1, 2], [3, 4], [5, 6]]
matrix_d = [[7, 8], [9, 10]]
print("Matrix C:")
for row in matrix_c:
    print(row)
print("\nMatrix D:")
for row in matrix_d:
    print(row)

print("Matrix Addition")
print("A+B : ")
Addition = matrix_addition(matrix_a, matrix_b)
for row in Addition:
    print(row)

print("Matrix Subtraction")
print("A-B : ")
Subtraction = matrix_subtraction(matrix_a, matrix_b)
for row in Subtraction:
    print(row)

print("Multiplication")
# Scalar Multiplication
print("2*A : ")
ScalarMultiplication = scalar_multiplication(matrix_a, 2)
for row in ScalarMultiplication:
    print(row)

# Matrix Multiplication
print("C*D :")
Multiplication = matrix_multiplication(matrix_c, matrix_d)
for row in Multiplication:
    print(row)

# Transpose
print("Transpose of D :")
Transpose = matrix_transpose(matrix_d)
for row in Transpose:
    print(row)
