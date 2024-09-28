import numpy as np
import cv2
def corner_sums(matrix, m, n,mask):
    # Top-left corner sum
    top_left = (matrix[0][0]*mask[0][0] + matrix[0][1]*mask[0][1] + matrix[1][0]*mask[1][0] + 
                matrix[1][1]*mask[1][1] + matrix[m-1][1]*mask[2][0] + matrix[1][n-1]*mask[0][2] + 
                matrix[m-1][n-2]*mask[2][1] + matrix[m-2][n-1]*mask[1][2] + matrix[m-1][n-1]*mask[2][2]
    )
    
    # Top-right corner sum
    top_right = (matrix[0][n-2]*mask[0][1] + matrix[0][n-1]*mask[0][2] + matrix[1][n-2]*mask[1][1] + 
                 matrix[1][n-1]*mask[1][2] + matrix[m-3][0]*mask[0][0] + matrix[m-2][0]*mask[1][0] + 
                 matrix[m-1][0]*mask[2][0] + matrix[m-1][1]*mask[2][1] + matrix[m-1][2]*mask[2][2]
    )
    
    # Bottom-left corner sum
    bottom_left =  (matrix[m-2][0]*mask[1][0] + matrix[m-2][1]*mask[1][1] + matrix[m-1][0]*mask[2][0] + 
                    matrix[m-1][1]*mask[2][1] + matrix[0][n-3]*mask[0][0] + matrix[0][n-2]*mask[0][1] + 
                    matrix[0][n-1]*mask[0][2] + matrix[1][n-1]*mask[1][2] + matrix[2][n-1]*mask[2][2]
    )
    
    # Bottom-right corner sum
    bottom_right = (matrix[m-2][n-2]*mask[1][1] + matrix[m-2][n-1]*mask[1][2] + matrix[m-1][n-2]*mask[2][1] + 
                    matrix[m-1][n-1]*mask[2][2] + matrix[0][0]*mask[0][0] + matrix[0][1]*mask[0][1] + 
                    matrix[0][2]*mask[0][2] + matrix[1][0]*mask[1][0] + matrix[2][0]*mask[2][0]
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

    corners = corner_sums(matrix, m, n,mask)

    # Assign corner sums to the result matrix
    result[0, 0] = corners["TL"]
    result[0, n-1] = corners["TR"]
    result[m-1, 0] = corners["BL"]
    result[m-1, n-1] = corners["BR"]

    for i in range(m):
        for j in range(n):
            # Leave corners already set
            if (i == 0 and j == 0) or (i == 0 and j == n-1) or (i == m-1 and j == 0) or (i == m-1 and j == n-1):
                continue
            # Top row 
            elif (i == 0) and (1 <= j <= n-2):
                    result[i, j] = (
                        matrix[0, j-1]*mask[0,0] + matrix[0, j]*mask[0,1] + matrix[0, j+1]*mask[0,2] +  
                        matrix[1, j-1]*mask[1,0] + matrix[1, j]*mask[1,1] + matrix[1, j+1]*mask[1,2] +  
                        matrix[m-1, j-1]*mask[2,0] + matrix[m-1, j]*mask[2,1] + matrix[m-1, j+1]*mask[2,2]  
                    )
            # Bottom row 
            elif (i == m-1) and (1 <= j <= n-2):
                    result[i, j] = (
                        matrix[0, j-1]*mask[0,0] + matrix[0, j]*mask[0,1] + matrix[0, j+1]*mask[0,2]  +
                        matrix[m-2, j-1]*mask[1,0] + matrix[m-2, j]*mask[1,1] + matrix[m-2, j+1]*mask[1,2] +
                        matrix[m-1, j-1]*mask[2,0] + matrix[m-1, j]*mask[2,1] + matrix[m-1, j+1]*mask[2,2]
                    )
            # Left column 
            elif (j == 0) and (1 <= i <= m-2):
                    result[i, j] = (
                        matrix[i+1, j]*mask[0,0] + matrix[i+1, 1]*mask[0,1] + matrix[i+1, n-1]*mask[0,2] +
                        matrix[i, j]*mask[1,0] + matrix[i, 1]*mask[1,1] + matrix[i, n-1]*mask[1,2] +
                        matrix[i-1, j]*mask[2,0] + matrix[i-1, 1]*mask[2,1] + matrix[i-1, n-1]*mask[2,2]
                    )
            # Right column 
            elif (j == n-1) and (1 <= i <= m-2):
                    result[i, j] = (
                        matrix[i-1, 0]*mask[0,0] + matrix[i-1, j-1]*mask[0,1] + matrix[i-1, j]*mask[0,2] +
                        matrix[i, 0]*mask[1,0] + matrix[i, j-1]*mask[1,1] + matrix[i, j]*mask[1,2] + 
                        matrix[i+1, 0]*mask[2,0] + matrix[i+1, j-1]*mask[2,1] + matrix[i+1, j]*mask[2,2]
                    )
            # Middle area based on the Averaging Filter Algorithm
            else:
                result[i, j] = (
                matrix[i-1, j-1] * mask[0, 0] + matrix[i-1, j] * mask[0, 1] + matrix[i-1, j+1] * mask[0, 2] +
                matrix[i, j-1] * mask[1, 0] + matrix[i, j] * mask[1, 1] + matrix[i, j+1] * mask[1, 2] +
                matrix[i+1, j-1] * mask[2, 0] + matrix[i+1, j] * mask[2, 1] + matrix[i+1, j+1] * mask[2, 2]
            )
    return result 
##########################################
#### REMOVE THIS TEST WHEN UPLOADING  ####
##########################################
mx = np.array([
    [7, 4, 0, 1],
    [5, 6, 2, 2],
    [6, 10, 7, 8],
    [1, 4, 2, 0],
    
])
testMatrix = side_sum_with_corners(mx)
print(testMatrix)
##########################################

# Read the image
img = cv2.imread('noisysalterpepper.png', cv2.IMREAD_GRAYSCALE)

# Check if image was loaded successfully
if img is None:
    raise ValueError("Image not found or unable to load.")

# Calculate side sums with corner sums
result_matrix = side_sum_with_corners(img)

# Assuming img and img_new are your images
img_new = result_matrix.astype(np.uint8)

# Concatenate the images horizontally
concatenated = np.hstack((img, img_new))

# Save and display image
cv2.imwrite('newSideCorner.png', img_new)
cv2.imwrite('concatenated.png',concatenated)
cv2.imshow('Original and New Image', concatenated)
cv2.waitKey(0)
cv2.destroyAllWindows()