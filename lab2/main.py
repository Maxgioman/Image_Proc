import cv2
import matplotlib.pyplot as plt
import numpy as np


def BFMatcher(des1, des2):
    brute_force = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    return sorted(brute_force.match(des1, des2), key=lambda x: x.distance)


def display(pic1, kpt1, pic2, kpt2, des1, des2):
    matches_count = BFMatcher(des1, des2)
    output_image = cv2.drawMatches(pic1, kpt1, pic2, kpt2, matches_count, None, flags=2)
    print(f'Total Number of Features matches found are {len(matches_count)}')
    plt.figure(figsize=(10, 20))
    plt.imshow(output_image)


def custom_match(kps1, descs1, kps2, descs2, img1, img2):
    matches = []
    for i in range(len(descs1)):
        for j in range(len(descs2)):
            distance = np.linalg.norm(descs1[i] - descs2[j])
            if distance < 850:
                matches.append((distance, i, j))

    match_to_draw = []
    for match in matches:
        match_to_draw.append(cv2.DMatch(_distance=match[0], _imgIdx=0, _queryIdx=match[1], _trainIdx=match[2]))

    img_with_matches = cv2.drawMatches(img1, kps1, img2, kps2, match_to_draw, None, flags=2)
    plt.figure(figsize=(10, 20))
    plt.imshow(img_with_matches)
    plt.show()


def low_contrast():
    img1 = cv2.imread('images/low_contrast.jpg')
    img2 = cv2.imread('images/low_contrast_dis.jpg')

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    detector = cv2.AKAZE_create()
    (kps1, descs1) = detector.detectAndCompute(gray1, None)
    (kps2, descs2) = detector.detectAndCompute(gray2, None)

    plt.imshow(cv2.drawKeypoints(gray1, kps1, img1))
    plt.imshow(cv2.drawKeypoints(gray2, kps2, img2))

    display(gray1, kps1, gray2, kps2, descs1, descs2)

    custom_match(kps1, descs1, kps2, descs2, gray1, gray2)


def high_contrast():
    img1 = cv2.imread('images/hign_contrast.jpg')
    img2 = cv2.imread('images/hign_contrast_dis.jpg')

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    detector = cv2.AKAZE_create()
    (kps1, descs1) = detector.detectAndCompute(gray1, None)
    (kps2, descs2) = detector.detectAndCompute(gray2, None)

    display(gray1, kps1, gray2, kps2, descs1, descs2)

    custom_match(kps1, descs1, kps2, descs2, gray1, gray2)


if __name__ == '__main__':
    mode = 1
    if mode == 0:
        high_contrast()
        low_contrast()
    elif mode == 1:
        high_contrast()
    elif mode == 2:
        low_contrast()
    else:
        print('Pick mode')
