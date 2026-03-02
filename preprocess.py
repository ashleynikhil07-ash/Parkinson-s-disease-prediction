import cv2
import numpy as np

image_path = r"C:\NIKHIL\SEM - 4\hackathon\dataset\parkinson\V01PE02.png"

img = cv2.imread(image_path)

if img is None:
    print("❌ Error: Image not found")
else:
    print("✅ Image loaded successfully")

    img = cv2.resize(img, (300, 300))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 100, 200)


    edge_count = np.sum(edges > 0)

    total_pixels = edges.size

    edge_density = edge_count / total_pixels

    print("Edge Count:", edge_count)
    print("Edge Density:", edge_density)

    if edge_density > 0.05:
        print("Possible Parkinson's Pattern")
    else:
        print("Normal Pattern")


    cv2.imshow("Original Image", img)
    cv2.imshow("Grayscale Image", gray)
    cv2.imshow("Edge Detection", edges)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("✅ Feature extraction completed")
