import cv2 as cv
import numpy as np

# read from video
vid = cv.VideoCapture('./resourses/video.MOV') #Here should go ur video name or if u gonna use ur cam, set to 0

# check if video is open
if (vid.isOpened() == False):
    print("Error opening video")


while (vid.isOpened()):
    # get status and frame form video
    ret, frame = vid.read()

    # if everything its okay, start processing
    if ret == True:
        # I need to scale bc the video i use its to big and dont fit in the normal screen
        # then i scaled to half
        scaled = cv.resize(frame, (0, 0), fx=0.5, fy=0.5,
                           interpolation=cv.INTER_LINEAR)

        # Now, bc the video was recorded in a room with yellow walls, if i use SHV colors, my skin
        # color was cofunsed with the backgorund walls, for that i have to use grayScales
        # and using a theory i call, "nearest object to the camera its more bright haha" i can now
        # detect the skin color, but only the one its more closes to the camera
        grays = cv.cvtColor(scaled, cv.COLOR_BGR2GRAY)
        # From transformation to grays scale, delimit the range of values to get
        skinRegionG = cv.inRange(grays, 190, 255)
        # Then we use GaussianBlur to fix a little bit the image
        blur = cv.GaussianBlur(skinRegionG, (5, 5), 0)
        # Now we search for our threshold in the same range
        ret, thresh = cv.threshold(
            blur, 185, 255, 0)
        # We find the contours
        contours, hierachy = cv.findContours(
            thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        # Next we need to find the biggest contourn
        contour = max(contours, key=lambda x: cv.contourArea(x))
        # We get values of coordinates and withd and height
        x, y, w, h = cv.boundingRect(contour)
        # This is for calulate the aspect ratio of the video
        width = vid.get(3)
        height = vid.get(4)
        # bc is scaled divided over the ratio
        lim = (int(width/2), int(height/4))
        # And we check, if the max contour with the cordinates overpass the limit of the square
        # we gonna draw in the video, we not detect the object, other wise, we draw a rectagle
        # and set the label to "Mano"
        if (y+h > lim[1]):
            pass
        else:
            cv.rectangle(scaled, (x, y), (x + w, y + h), (0, 255, 255), 0)
            cv.putText(scaled, "Mano", (x, y+h+20),
                       cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        # put rectangle to get area of detection
        cv.rectangle(scaled, (0, 0), lim, (0, 0, 255), 2)
        # Put text to the area
        cv.putText(scaled, "Area deteccion", (0, 340),
                   cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        # display video, to see the deteccion in acction
        cv.imshow("Video", scaled)

        if cv.waitKey(25) & 0xFF == ord('q'):
            break
    # Just to check if the frame dont crash or something happend
    else:
        print("Problem reading frame")
        break

# Once we finish we realize de memory of video
vid.release()
# And destroy all windows created
cv.destroyAllWindows()
