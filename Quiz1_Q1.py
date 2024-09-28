import numpy as np
import cv2
import numpy as np
import cv2
def corner_sums(matrix, m, n):
    # Convert matrix elements to a larger data type (e.g., int64)
    matrix = matrix.astype(np.int64)

    # Top-left corner sum
    top_left = (matrix[0][0] + matrix[0][1] + matrix[1][0] + 
                matrix[1][1] + matrix[m-1][1] + matrix[1][n-1] + 
                matrix[m-1][n-2] + matrix[m-2][n-1] + matrix[m-1][n-1]
    )
    
    # Top-right corner sum
    top_right = (matrix[0][n-2] + matrix[0][n-1] + matrix[1][n-2] + 
                 matrix[1][n-1] + matrix[m-3][0] + matrix[m-2][0] + 
                 matrix[m-1][0] + matrix[m-1][1] + matrix[m-1][2]
    )
    
    # Bottom-left corner sum
    bottom_left =  (matrix[m-2][0] + matrix[m-2][1] + matrix[m-1][0] + 
                    matrix[m-1][1] + matrix[0][n-3] + matrix[0][n-2] + 
                    matrix[0][n-1] + matrix[1][n-1] + matrix[2][n-1]
    )
    
    # Bottom-right corner sum
    bottom_right = (matrix[m-2][n-2] + matrix[m-2][n-1] + matrix[m-1][n-2] + 
                    matrix[m-1][n-1] + matrix[0][0] + matrix[0][1] + 
                    matrix[0][2] + matrix[1][0] + matrix[2][0]
    )
    
    return {
        "TL": top_left,
        "TR": top_right,
        "BL": bottom_left,
        "BR": bottom_right
    }
def side_sum_with_corners(matrix):
    m, n = matrix.shape  # Get dimensions of the matrix
    result = np.zeros((m, n))  # Initialize a result matrix of the same size
    mask = np.ones([3, 3], dtype=int) / 9  # 3x3 averaging mask

    corners = corner_sums(matrix, m, n)

    # Assign corner sums to the result matrix
    result[0, 0] = corners["TL"]
    result[0, n-1] = corners["TR"]
    result[m-1, 0] = corners["BL"]
    result[m-1, n-1] = corners["BR"]

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