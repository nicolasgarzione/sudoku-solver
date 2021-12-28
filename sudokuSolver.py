'''from ctypes import cdll
lib = cdll.LoadLibrary('./solve.o')

class Solve(object):
    def __init__(self):
        self.obj = lib.Solve_new()

    def sudokuSolver(self):
        lib.Solve_sudokuSolver(self.obj)

grid =  [[0,4,3,0,8,0,2,5,0],
         [6,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,1,0,9,4],
         [9,0,0,0,0,4,0,7,0],
         [0,0,0,6,0,8,0,0,0],
         [0,1,0,2,0,0,0,0,3],
         [8,2,0,5,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,5],
         [0,3,4,0,9,0,7,1,0]]

s = Solve()
s.grid = grid
solved = s.sudokuSolver(s.grid)'''

import numpy as np
import cv2 as cv
import matplotlib



def main():
    cap = cv.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        grayscale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        blur = cv.GaussianBlur(grayscale, (1,1), cv.BORDER_DEFAULT)

        threshold = cv.adaptiveThreshold(blur,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY_INV,199,5)

        contours, hierarchy = cv.findContours(threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        #print("Number of Contours found = " + str(len(contours)))

        largest = 0
        i = 0
        index = 0

        for c in contours:
            area = cv.contourArea(c)
            if area > largest:
                largest = area
                index = i
            i += 1

        contour = contours[index]

        epsilon = 0.01 * cv.arcLength(contour, True)

        approxShape = cv.approxPolyDP(contour, epsilon, True)

        
        if len(approxShape) == 4:
            cv.drawContours(frame, [approxShape], 0, (0,255,0), 3)
            n_shape = findCorners(approxShape)
            n_size = 900
            n_window = np.array([[0,n_size],[n_size,n_size],[n_size,0],[0,0]],np.float32)
            M = cv.getPerspectiveTransform(n_shape, n_window)
            out = cv.warpPerspective(grayscale, M, (n_size, n_size))
            sudoku_unsolved = findNumbers(out)
            #cv.imshow('Adjusted Image', out)

        cv.imshow('Image', frame)

        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


def findCorners(m):
    x = [m[0,0,0], m[1,0,0], m[2,0,0], m[3,0,0]]
    y = [m[0,0,1], m[1,0,1], m[2,0,1], m[3,0,1]]

    xmin1 = min(x)
    x1_i = x.index(xmin1)
    x[x1_i] = 10000000
    xmin2 = min(x)
    x2_i = x.index(xmin2)
    x[x1_i] = xmin1

    if y[x1_i]>y[x2_i]:
        top_left = x2_i
        bottom_left = x1_i
    else:
        top_left = x1_i
        bottom_left = x2_i

    temp = [0, 1, 2, 3]
    id = []
    for i in temp:
        if i not in [top_left, bottom_left]:
            id.append(i)

    if y[id[0]]>y[id[1]]:
        top_right = id[1]
        bottom_right = id[0]
    else:
        top_right = id[0]
        bottom_right = id[1]

    return np.array([m[bottom_left,:,:], m[bottom_right,:,:], m[top_right,:,:], m[top_left,:,:]],np.float32)

def findNumbers(im):
    N = 100
    tile = [im[x:x+N,y:y+N] for x in range(0,im.shape[0],N) for y in range(0,im.shape[1],N)]
    thresh = []
    num_labels = []
    labels = []
    stats = []
    centroids = []
    i = 0
    for t in tile:
        thresh[i] = cv.threshold(t,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        output = cv.connectedComponentsWithStats(thresh[i], 4, cv.CV_32S)
        num_labels[i] = output[0]
        labels[i] = output[1]
        stats[i] = output[2]
        centroids[i] = output[3]
        i += 1
    
    #tile[0] = cv.threshold(tile[0],0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    #cv.imshow('Tile', tile[0])
    return 

if __name__ == '__main__':
    main()
