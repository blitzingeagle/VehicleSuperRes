import cv2

filename = "videos/cable00_detected.avi"
output_dir = "frames"

vidcap = cv2.VideoCapture(filename)
success, image = vidcap.read()
count = 1

while success:
    cv2.imwrite("{}/frame{:07d}.jpg".format(output_dir, count), image)
    print(image.shape)

    cv2.imshow('frame', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    success, image = vidcap.read()
    count += 1