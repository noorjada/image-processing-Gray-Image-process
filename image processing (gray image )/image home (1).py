import cv2
import numpy as np

# Task 1: Implement component labeling algorithm with 4-connectivity
def label_components_4_connectivity(image):
    height, width = image.shape
    labeled_image = np.zeros((height, width), dtype=np.int32)
    label = 1
    
    def flood_fill(x, y, label):
        stack = [(x, y)]
        while stack:
            x, y = stack.pop()
            if x < 0 or x >= height or y < 0 or y >= width:
                continue
            if image[x, y] == 255 and labeled_image[x, y] == 0:
                labeled_image[x, y] = label
                stack.extend([(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)])
    
    for i in range(height):
        for j in range(width):
            if image[i, j] == 255 and labeled_image[i, j] == 0:
                flood_fill(i, j, label)
                label += 1
                
    return labeled_image

# Task 2: Modify the algorithm to consider 8-connectivity
def label_components_8_connectivity(image):
    height, width = image.shape
    labeled_image = np.zeros((height, width), dtype=np.int32)
    label = 1
    
    def flood_fill(x, y, label):
        stack = [(x, y)]
        while stack:
            x, y = stack.pop()
            if x < 0 or x >= height or y < 0 or y >= width:
                continue
            if image[x, y] == 255 and labeled_image[x, y] == 0:
                labeled_image[x, y] = label
                stack.extend([(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1),
                              (x - 1, y - 1), (x - 1, y + 1), (x + 1, y - 1), (x + 1, y + 1)])
    
    for i in range(height):
        for j in range(width):
            if image[i, j] == 255 and labeled_image[i, j] == 0:
                flood_fill(i, j, label)
                label += 1
                
    return labeled_image

# Task 3: Modify the algorithm to consider a range of intensity values within the set V
def label_components_intensity_range(image, min_val, max_val):
    height, width = image.shape
    labeled_image = np.zeros((height, width), dtype=np.int32)
    label = 1
    
    def flood_fill(x, y, label):
        stack = [(x, y)]
        while stack:
            x, y = stack.pop()
            if x < 0 or x >= height or y < 0 or y >= width:
                continue
            if min_val <= image[x, y] <= max_val and labeled_image[x, y] == 0:
                labeled_image[x, y] = label
                stack.extend([(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1),
                              (x - 1, y - 1), (x - 1, y + 1), (x + 1, y - 1), (x + 1, y + 1)])
    
    for i in range(height):
        for j in range(width):
            if min_val <= image[i, j] <= max_val and labeled_image[i, j] == 0:
                flood_fill(i, j, label)
                label += 1
                
    return labeled_image

# Task 4: Implement the size filter algorithm
def size_filter(labeled_image, min_size, max_size):
    unique, counts = np.unique(labeled_image, return_counts=True)
    filtered_image = np.zeros_like(labeled_image)
    
    for label, count in zip(unique, counts):
        if min_size <= count <= max_size:
            filtered_image[labeled_image == label] = label
    
    return filtered_image

# Load images
binary_image = cv2.imread("F:/imagee.png", cv2.IMREAD_GRAYSCALE)
gray_image = cv2.imread("F:/nnn.jpg", cv2.IMREAD_GRAYSCALE)

# Binarize the binary_image for Task 1, 2, 4
_, binary_image = cv2.threshold(binary_image, 127, 255, cv2.THRESH_BINARY)

# Task 1: 4-connectivity
labeled_4 = label_components_4_connectivity(binary_image)
print("4-connectivity labeled image:\n", labeled_4)

# Task 2: 8-connectivity
labeled_8 = label_components_8_connectivity(binary_image)
print("8-connectivity labeled image:\n", labeled_8)

# Task 3: Intensity range connectivity
min_val = int(input("Enter minimum intensity value: "))
max_val = int(input("Enter maximum intensity value: "))
labeled_intensity = label_components_intensity_range(gray_image, min_val, max_val)
print("Intensity range labeled image:\n", labeled_intensity)

# Task 4: Size filter
min_size = int(input("Enter minimum size: "))
max_size = int(input("Enter maximum size: "))
filtered_image = size_filter(labeled_8, min_size, max_size)
print("Size filtered image:\n", filtered_image)

# Display results
def colorize_labels(labeled_image):
    colors = np.random.randint(0, 255, (np.max(labeled_image) + 1, 3))
    colorized_image = np.zeros((labeled_image.shape[0], labeled_image.shape[1], 3), dtype=np.uint8)
    for label in range(1, np.max(labeled_image) + 1):
        colorized_image[labeled_image == label] = colors[label]
    return colorized_image

colorized_labeled_4 = colorize_labels(labeled_4)
colorized_labeled_8 = colorize_labels(labeled_8)

# Add a pink border to match the provided output example
pink_border = [255, 182, 193]
border_size = 50

output_image_4 = cv2.copyMakeBorder(colorized_labeled_4, border_size, border_size, border_size, border_size, 
                                    cv2.BORDER_CONSTANT, value=pink_border)
output_image_8 = cv2.copyMakeBorder(colorized_labeled_8, border_size, border_size, border_size, border_size, 
                                    cv2.BORDER_CONSTANT, value=pink_border)

cv2.imshow("4-connectivity", output_image_4)
cv2.imshow("8-connectivity", output_image_8)
cv2.imshow("Intensity range", labeled_intensity.astype(np.uint8) * (255 // labeled_intensity.max()))
cv2.imshow("Size filtered", filtered_image.astype(np.uint8) * (255 // filtered_image.max()))
cv2.waitKey(0)
cv2.destroyAllWindows()
