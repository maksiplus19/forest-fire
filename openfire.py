#   This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.	

import os
from functools import reduce
from typing import List, Dict, Tuple

import cv2
import numpy as np
import skimage.color as color
import skimage.segmentation as seg


class FireAlgorithm:
    DATASET = '../wildfire dataset/test'
    _precomputed_des = None

    @staticmethod
    def _precompute_descriptors(detector):
        # проверяем были ли уже вычеслины дескрипторы, если да, то сразу возвращаем
        if FireAlgorithm._precomputed_des is not None:
            return FireAlgorithm._precomputed_des
        test_descriptors = []
        for t in os.listdir(FireAlgorithm.DATASET):
            tmp = cv2.imread(f'{FireAlgorithm.DATASET}/{t}', cv2.COLOR_BGR2GRAY)
            res = detector.detectAndCompute(tmp, None)
            if res[1] is None:
                continue
            test_descriptors.append(res)
        FireAlgorithm._precomputed_des = test_descriptors.copy()
        return test_descriptors

    @staticmethod
    def detect_fire(img: np.ndarray) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        # небольшое повышение контрастности
        img = cv2.convertScaleAbs(img, alpha=1.1, beta=-10)

        # пороговая сегментация
        mask = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 170, 255, cv2.THRESH_BINARY)[1]
        # применение эрозии и дилатации для удаления мельких деталей
        k = cv2.getStructuringElement(cv2.MORPH_ERODE, (10, 10))
        processed = cv2.erode(mask, k)
        k = cv2.getStructuringElement(cv2.MORPH_DILATE, (10, 10))
        processed = cv2.dilate(processed, k)
        processed = cv2.dilate(processed, k)
        processed[processed > 0] = 2
        # выделение фоновой части изображения
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
        processed_inv = cv2.dilate(mask, k)
        processed_inv = cv2.bitwise_not(processed_inv)
        processed_inv[processed_inv > 0] = 1

        # сегментация на базе морфологического водораздела
        img_seg = seg.watershed(color.rgb2gray(img), processed + processed_inv)
        mask = img_seg == 2
        img_seg[mask] = 255
        img_seg[~mask] = 0
        img_seg = img_seg.astype(np.uint8)

        # выделение и фильтрация контуров
        contours, _ = cv2.findContours(img_seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        detected_contours = []
        for contour in contours:
            if cv2.contourArea(contour) > 0.9 * img.shape[0] * img.shape[1] or \
                    cv2.contourArea(contour) < 400:
                continue
            contour = cv2.boundingRect(contour)
            l_t = (contour[0], contour[1])
            r_b = (contour[0] + contour[2], contour[1] + contour[3])
            detected_contours.append((l_t, r_b))

        # расчет дескрипторов и поиск совпадений
        detector = cv2.SIFT_create()
        matcher = cv2.BFMatcher_create()
        test_descriptors = FireAlgorithm._precompute_descriptors(detector)
        detected_fire = []
        for k, (lt_pair, rb_pair) in enumerate(detected_contours):
            part = img[lt_pair[1]:rb_pair[1], lt_pair[0]:rb_pair[0]]
            gray = cv2.cvtColor(part, cv2.COLOR_BGR2RGB)
            kp, desc = detector.detectAndCompute(gray, None)
            matches = [matcher.knnMatch(desc, d[1], k=2) for d in test_descriptors]

            for i in range(len(matches)):
                good = []
                for m, n in matches[i]:
                    if m.distance < 0.7 * n.distance:
                        good.append(m)
                matches[i] = good

            matches = reduce(lambda a, b: a + b, matches)
            if len(matches) > 1:
                detected_fire.append((lt_pair, rb_pair))
        return detected_fire

    @staticmethod
    def detect_burned_area(img: np.ndarray) -> np.ndarray:
        low = np.array([100, 0, 0])
        up = np.array([140, 255, 132])

        # пороговая сегментация
        mask = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), low, up)

        # удаление мелких деталей и создание маски фона
        k = cv2.getStructuringElement(cv2.MORPH_ERODE, (30, 30))
        processed = cv2.erode(mask, k)
        k = cv2.getStructuringElement(cv2.MORPH_DILATE, (100, 100))
        processed_inv = cv2.dilate(mask, k)
        processed_inv = cv2.dilate(processed_inv, k)
        k = cv2.getStructuringElement(cv2.MORPH_DILATE, (10, 10))
        processed_inv = cv2.dilate(processed_inv, k)
        processed_inv = cv2.bitwise_not(processed_inv)

        # объединение меток
        processed_inv[processed_inv != 0] = 2
        processed += processed_inv

        # расчет и возрат результата
        return seg.random_walker(img, processed, multichannel=True)

    @staticmethod
    def calculate_characteristics(img: np.ndarray) -> Dict:
        characteristics = {}
        # выделение площади пройденой огнем
        burned = FireAlgorithm.detect_burned_area(img)
        # выделение контуров и расчет характеристик на их основе
        contours, _ = cv2.findContours(burned, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        characteristics['burned_area'] = sum(map(cv2.contourArea, contours))
        characteristics['perimeter'] = sum(map(lambda x: cv2.arcLength(x, True), contours))
        characteristics['coordinates'] = []
        for c in contours:
            contour = cv2.boundingRect(c)
            l_t = (contour[0], contour[1])
            r_b = (contour[0] + contour[2], contour[1] + contour[3])
            characteristics['coordinates'].append((l_t, r_b))
        return characteristics

    @staticmethod
    def predict(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        # расчет минимальных описывающих окружностей
        contours, _ = cv2.findContours(img1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        _, radius1 = cv2.minEnclosingCircle(contours[0])
        contours, _ = cv2.findContours(img2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        _, radius2 = cv2.minEnclosingCircle(contours[0])

        # предположение
        side = 2 * int(radius2 - radius1)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (side, side))
        return cv2.dilate(img2, k)
