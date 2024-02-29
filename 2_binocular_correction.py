import numpy as np
import cv2

def load_calibration_data(left_calibration_path, right_calibration_path, stereo_calibration_path):
    # 加载单目校准数据
    left_data = np.load(left_calibration_path)
    right_data = np.load(right_calibration_path)
    
    # 加载立体校准数据
    stereo_data = np.load(stereo_calibration_path)
    
    return left_data['mtx'], left_data['dist'], right_data['mtx'], right_data['dist'], stereo_data['R'], stereo_data['T']

def stereo_rectify(mtx_left, dist_left, mtx_right, dist_right, R, T, image_size, alpha=1):
    # 计算校正变换
    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
        mtx_left, dist_left, mtx_right, dist_right, image_size, R, T, alpha)
    return R1, R2, P1, P2, Q

def init_undistort_rectify_map(mtx, dist, R, P, image_size):
    # 计算畸变校正和立体校正的映射矩阵
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, R, P, image_size, cv2.CV_32FC1)
    return mapx, mapy

def rectify_images(left_img_path, right_img_path, map1x, map1y, map2x, map2y):
    # 读取图像
    left_img = cv2.imread(left_img_path)
    right_img = cv2.imread(right_img_path)
    
    # 应用映射进行畸变校正和立体校正
    rectified_left = cv2.remap(left_img, map1x, map1y, cv2.INTER_LINEAR)
    rectified_right = cv2.remap(right_img, map2x, map2y, cv2.INTER_LINEAR)
    
    return rectified_left, rectified_right

def main():
    left_calibration_path = 'calibration_data_left.npz'
    right_calibration_path = 'calibration_data_right.npz'
    stereo_calibration_path = 'stereocali.npz'
    left_img_path = "testleft.bmp"
    right_img_path = "testright.bmp"
    image_size = (2048, 2448)
    
    # 加载校准数据
    mtx_left, dist_left, mtx_right, dist_right, R, T = load_calibration_data(
        left_calibration_path, right_calibration_path, stereo_calibration_path)
    
    # 计算立体校正参数
    R1, R2, P1, P2, Q = stereo_rectify(mtx_left, dist_left, mtx_right, dist_right, R, T, image_size)
    
    # 初始化畸变校正映射
    map1x, map1y = init_undistort_rectify_map(mtx_left, dist_left, R1, P1, image_size)
    map2x, map2y = init_undistort_rectify_map(mtx_right, dist_right, R2, P2, image_size)
    np.savez('calibration_maps.npz', map1x=map1x, map1y=map1y, map2x=map2x, map2y=map2y)
    
    # 校正图像
    rectified_left, rectified_right = rectify_images(left_img_path, right_img_path, map1x, map1y, map2x, map2y)
    
    # 保存和显示校正后的图像
    cv2.imwrite("rectified_left.bmp", rectified_left)
    cv2.imwrite("rectified_right.bmp", rectified_right)

    concat = cv2.hconcat([rectified_left, rectified_right])

    i = 0
    while (i < 2048):
        cv2.line(concat, (0,i), (4896,i), (0, 255, 0))
        i += 50
    print("Q:",Q)
    cv2.imwrite('rectified.bmp',concat)
    cv2.imshow("rectified", concat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    

if __name__ == "__main__":
    main()
