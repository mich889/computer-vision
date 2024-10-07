import cv2
import matplotlib.pyplot as plt

average_face_path = '/Users/michellechen/cs/classes/cs180/computer-vision/proj3/average_brazilian_face_rgb.jpg'
img1_path = '/Users/michellechen/cs/classes/cs180/computer-vision/proj3/images/img1-1.png'

# Load both images
average_face_image = cv2.imread(average_face_path)
img1 = cv2.imread(img1_path)

# Get the dimensions of the second image (img1)
img1_height, img1_width = img1.shape[:2]

# Resize the average face image to match the dimensions of img1
average_face_resized = cv2.resize(average_face_image, (img1_width, img1_height))

# Convert the resized average face image from BGR to RGB for display
average_face_resized_rgb = cv2.cvtColor(average_face_resized, cv2.COLOR_BGR2RGB)

# Display the resized average face image
plt.imshow(average_face_resized_rgb)
plt.title("Resized Average Brazilian Face")
plt.show()

# Save the resized image
resized_image_path = '/Users/michellechen/cs/classes/cs180/computer-vision/proj3/average_brazilian_face_resized.jpg'
cv2.imwrite(resized_image_path, average_face_resized)