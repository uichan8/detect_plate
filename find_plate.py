import cv2
import matplotlib.pyplot as plt
import numpy as np

weight = []
for i in range(10):
    img_path = "num/" + str(i) + ".png"
    digit = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    digit = cv2.resize(digit, (100,100), interpolation=cv2.INTER_LINEAR)
    digit = digit.reshape(-1)
    weight.append(digit)
weight = (np.array(weight)/255).T

def find_plate(img):
    # adaptiveThreshold
    th = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,301, 37)
    white = 255-(255 * (th > 100)).astype(np.uint8)

    #erode and dilate
    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(white, kernel, iterations=3)
    dilation = cv2.dilate(erosion, kernel, iterations=2)

    #find object
    num_labels, labelmap, stats, centers = cv2.connectedComponentsWithStats(dilation)

    #find number
    new_label = []
    for l in range(1, num_labels):
        x, y, width, height, pels = stats[l]
        if (0.2*height <= width <= 0.7*height) and width * height > 3000: #비율과 크기 확인
            crop = white[y:y+height,x:x+width]
            if(crop.mean() > 255 * 0.3 and crop.mean() < 255 * 0.6): #속채움 비율 확인(완전 하얀색이나 검은색인것을 배재)
                vector = cv2.resize(crop, (100,100), interpolation=cv2.INTER_LINEAR).reshape(-1,1)/255
                distance = ((weight - vector)**2).mean(axis = 0).max()
                if distance > 0.7: #번호판 글꼴과 비교(0부터 9까지 숫자랑 비교해서 일정수준 이상으로 비슷한 것이 있으면 번호 object)
                    new_label.append(stats[l])

    new_label = np.array(new_label)

    y1 = (new_label[:,0]).min()
    x1 = (new_label[:,1]).min()
    y2 = (new_label[:,0] + new_label[:,2]).max()
    x2 = (new_label[:,1] + new_label[:,3]).max()

    plate = img[x1-100:x2+100,y1-200:y2+200]
    return plate

        

if __name__ == "__main__":
    img = cv2.imread('001.jpg', cv2.IMREAD_GRAYSCALE)
    find_plate(img)