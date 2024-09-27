import numpy as np
import cv2
#WORKING LIKE A GOAT BUT SLOW 
def corner_sums(matrix):
    # Top-left corner sum
    top_left = sum([matrix[i][j] for i in range(2) for j in range(2)]) + matrix[3][1] + matrix[1][3] + matrix[3][2] + matrix[2][3] + matrix[3][3]
    
    # Top-right corner sum
    top_right = sum([matrix[i][j] for i in range(2) for j in range(2, 4)]) + matrix[1][0] + matrix[2][0] + matrix[3][0] + matrix[3][1] + matrix[3][2]
    
    # Bottom-left corner sum
    bottom_left = sum([matrix[i][j] for i in range(2, 4) for j in range(2)]) + sum(matrix[0][1:4]) + matrix[1][3] + matrix[2][3]
    
    # Bottom-right corner sum
    bottom_right = sum([matrix[i][j] for i in range(2, 4) for j in range(2, 4)]) + sum(matrix[0][:3]) + matrix[1][0] + matrix[2][0]
    
    return {
        "Top-left corner sum": top_left,
        "Top-right corner sum": top_right,
        "Bottom-left corner sum": bottom_left,
        "Bottom-right corner sum": bottom_right
    }

def side_sum_with_corners(matrix):
    m, n = matrix.shape  # Get dimensions of the matrix
    result = np.zeros((m, n))  # Initialize a result matrix of the same size
    mask = np.ones([3, 3], dtype=int) / 9  # 3x3 averaging mask

    # Calculate corner sums
    corners = corner_sums(matrix)

    # Assign corner sums to the result matrix
    result[0, 0] = corners["Top-left corner sum"]
    result[0, n-1] = corners["Top-right corner sum"]
    result[m-1, 0] = corners["Bottom-left corner sum"]
    result[m-1, n-1] = corners["Bottom-right corner sum"]

    for i in range(1, m-1):
        for j in range(1, n-1):
            # Multiply by the correct mask index for the center region
            result[i, j] = (
                matrix[i-1, j-1] * mask[0, 0] + matrix[i-1, j] * mask[0, 1] + matrix[i-1, j+1] * mask[0, 2] +
                matrix[i, j-1] * mask[1, 0] + matrix[i, j] * mask[1, 1] + matrix[i, j+1] * mask[1, 2] +
                matrix[i+1, j-1] * mask[2, 0] + matrix[i+1, j] * mask[2, 1] + matrix[i+1, j+1] * mask[2, 2]
            )

    # Handle the boundaries (top, bottom, left, right) similarly
    for i in range(m):
        for j in range(n):
            if (i == 0 and j == 0) or (i == 0 and j == n-1) or (i == m-1 and j == 0) or (i == m-1 and j == n-1):
                continue  # Skipping the corners as they are already calculated
            elif i == 0 and 1 <= j <= n-2:  # Top row
                result[i, j] = (
                    matrix[i, j-1] * mask[1, 0] + matrix[i, j] * mask[1, 1] + matrix[i, j+1] * mask[1, 2] +
                    matrix[i+1, j-1] * mask[2, 0] + matrix[i+1, j] * mask[2, 1] + matrix[i+1, j+1] * mask[2, 2]
                )
            elif i == m-1 and 1 <= j <= n-2:  # Bottom row
                result[i, j] = (
                    matrix[i-1, j-1] * mask[0, 0] + matrix[i-1, j] * mask[0, 1] + matrix[i-1, j+1] * mask[0, 2] +
                    matrix[i, j-1] * mask[1, 0] + matrix[i, j] * mask[1, 1] + matrix[i, j+1] * mask[1, 2]
                )
            elif j == 0 and 1 <= i <= m-2:  # Left column
                result[i, j] = (
                    matrix[i-1, j] * mask[0, 1] + matrix[i-1, j+1] * mask[0, 2] +
                    matrix[i, j] * mask[1, 1] + matrix[i, j+1] * mask[1, 2] +
                    matrix[i+1, j] * mask[2, 1] + matrix[i+1, j+1] * mask[2, 2]
                )
            elif j == n-1 and 1 <= i <= m-2:  # Right column
                result[i, j] = (
                    matrix[i-1, j-1] * mask[0, 0] + matrix[i-1, j] * mask[0, 1] +
                    matrix[i, j-1] * mask[1, 0] + matrix[i, j] * mask[1, 1] +
                    matrix[i+1, j-1] * mask[2, 0] + matrix[i+1, j] * mask[2, 1]
                )
    return result 


# Calculate side sums with corner sums


# Read the image
img = cv2.imread('noisysalterpepper.png', cv2.IMREAD_GRAYSCALE)

# Check if image was loaded successfully
if img is None:
    raise ValueError("Image not found or unable to load.")

# Calculate side sums with corner sums
result_matrix = side_sum_with_corners(img)

img_new = result_matrix.astype(np.uint8) 
cv2.imwrite('asdfg.png', img_new) 
cv2.imshow('Old Image', img)
cv2.imshow('New Image', img_new)
cv2.waitKey(0)
cv2.destroyAllWindows()