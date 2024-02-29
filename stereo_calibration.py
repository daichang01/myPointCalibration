import numpy as np
import cv2
import glob
import math

def load_calibration_data(left_calib_file, right_calib_file):
    # 从文件加载相机标定数据
    left_data = np.load(left_calib_file)
    right_data = np.load(right_calib_file)
    return left_data['mtx'], left_data['dist'], right_data['mtx'], right_data['dist']

def find_chessboard_corners(images_left, images_right, pattern_size, square_size):
    # 查找棋盘格角点
    obj_points = [] # 真实世界空间中的3d点
    img_points_left = [] # 图像平面中的2d点
    img_points_right = []
    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2) * square_size

    for img_left_path, img_right_path in zip(images_left, images_right):
        img_left = cv2.imread(img_left_path, cv2.IMREAD_GRAYSCALE)
        img_right = cv2.imread(img_right_path, cv2.IMREAD_GRAYSCALE)
        found_left, corners_left = cv2.findChessboardCorners(img_left, pattern_size)
        found_right, corners_right = cv2.findChessboardCorners(img_right, pattern_size)

        if found_left and found_right:
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
            cv2.cornerSubPix(img_left, corners_left, (5, 5), (-1, -1), term)
            cv2.cornerSubPix(img_right, corners_right, (5, 5), (-1, -1), term)

            img_points_left.append(corners_left.reshape(-1, 2))
            img_points_right.append(corners_right.reshape(-1, 2))
            obj_points.append(pattern_points)

    return obj_points, img_points_left, img_points_right

def stereo_calibrate_cameras(obj_points, img_points_left, img_points_right, mtx_left, dist_left, mtx_right, dist_right, image_size):
    # 对相机进行立体标定
    flags = cv2.CALIB_FIX_INTRINSIC
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    ret, mtx_left, dist_left, mtx_right, dist_right, R, T, E, F = cv2.stereoCalibrate(
        obj_points, img_points_left, img_points_right, mtx_left, dist_left, mtx_right, dist_right, image_size,
        flags=flags, criteria=criteria)
    return R, T

def calculate_camera_distance(T):
    # 计算相机之间的距离
    return math.sqrt(T[0]**2 + T[1]**2 + T[2]**2)

def main():
    square_size = 2.0 # 棋盘格方格的实际大小
    pattern_size = (11, 8) # 棋盘格模式大小
    images_left = glob.glob('dataset/left/*.png') # 左侧图像路径
    images_right = glob.glob('dataset/right/*.png') # 右侧图像路径
    mtx_left, dist_left, mtx_right, dist_right = load_calibration_data('calibration_data_left.npz', 'calibration_data_right.npz')
    
    # 获取图像大小
    sample_image = cv2.imread(images_left[0])
    image_size = sample_image.shape[::-1][1:] # (width, height)
    
    # 查找棋盘格角点
    obj_points, img_points_left, img_points_right = find_chessboard_corners(images_left, images_right, pattern_size, square_size)
    # 进行立体标定
    R, T = stereo_calibrate_cameras(obj_points, img_points_left, img_points_right, mtx_left, dist_left, mtx_right, dist_right, image_size)
    
    # 计算相机距离
    distance = calculate_camera_distance(T)
    print("Camera distance:", distance, "cm")
    print("R:",R)
    np.savez("stereocali.npz", R=R, T=T)


if __name__ == "__main__":
    main()
