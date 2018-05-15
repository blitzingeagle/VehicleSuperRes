import cv2

from commands import getoutput as gop

dirname = 'picture'
filenames = gop('ls ' + dirname).split('\n')

count = 0

for filename in filenames:
    image = cv2.imread(dirname + '/' + filename)

    # cv2.imshow('rt',image)
    # cv2.waitKey(0)

    if count == 0:
        height, width, channel = image.shape
        print
        height, width

        video = cv2.VideoWriter('test.avi', cv2.VideoWriter_fourcc(*"XVID"), 100, (width, height))

    video.write(image)
    print(count)
    count += 1

video.release()