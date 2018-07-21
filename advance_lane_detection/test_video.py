import cv2
import os

def extractFrames(pathIn):

    cap = cv2.VideoCapture(pathIn)
    count = 0

    while (cap.isOpened()):
        print ("file is opened")
        # Capture frame-by-frame
        ret, frame = cap.read()
        print (ret)
        if ret == True:
            print('Read %d frame: ' % count, ret)
            cv2.imshow("frame", frame)
            cv2.waitKey(25)
            count += 1

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def main():
    extractFrames('../project_video.mp4')

if __name__=="__main__":
    main()
