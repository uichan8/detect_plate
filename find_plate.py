import cv2
import matplotlib.pyplot as plt
import numpy as np

def find_plate(img):
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,101, 37)
    white = 255-(255 * (img > 100)).astype(np.uint8)

    #edges = white
    edges = cv2.Canny(white, 100, 200,L2gradient=True)
    connectivity = 100
    num_labels, labelmap, stats, centers = cv2.connectedComponentsWithStats(edges, connectivity, cv2.CV_32S)

    map = np.zeros_like(img)
    map2 = np.ones_like(img)*255
    avg_w, avg_h, avg_y, n = 0, 0, 0, 0
    new_label = []
    for l in range(1, num_labels):
        x, y, width, height, pels = stats[l]
        if (0.2*height <= width <= 0.7*height) and width * height> 7000 : #비율과 크기 확인
            crop = white[y:y+height,x:x+width]
            th = 0.3
            if(crop.mean() > 255 * th and crop.mean() < 255 * (1-th)):
                map[labelmap==l] = 255
                map2[y:y+height,x:x+width] = 255 - crop
                avg_w += width
                avg_h += height
                avg_y += y
                n += 1
                new_label.append(stats[l])

    new_label = np.array(new_label)

    

    x_argmax = new_label[:,0].argmax()
    x_argmin = new_label[:,0].argmin()

    lb = new_label[x_argmin,:2]
    rt = new_label[x_argmax,:2] + new_label[x_argmax,2:4]

    result = white[lb[1]:rt[1],lb[0]:rt[0]]

    return result
        


        

if __name__ == "__main__":
    img = cv2.imread('001.jpg', cv2.IMREAD_GRAYSCALE)
    find_plate(img)