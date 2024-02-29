import numpy as np
import cv2
import glob

def calibrate_camera(square_size, width, height, image_folder):
    objp = np.zeros((width*height, 3), np.float32)
    # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y
    objp[:,:2] = np.mgrid[0:width,0:height].T.reshape(-1,2) * square_size

    # 保存角点的数组
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    # 读取图像文件

    images = glob.glob(f'{image_folder}/*.bmp')

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 查找棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, (width,height), None)

        # 如果找到了，添加物理坐标和图像坐标
        if ret == True:
            objpoints.append(objp) #物理坐标的z都为0
            corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)

            # 可以选择绘制并显示角点
            cv2.drawChessboardCorners(img, (width,height), corners2, ret)
            cv2.namedWindow('img', cv2.WINDOW_NORMAL)
            cv2.imshow('img', img)
            cv2.waitKey(200)

    cv2.destroyAllWindows()

    # 标定
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    print("Camera matrix : \n", mtx)
    print("dist : \n", dist)
    print("rvecs : \n", rvecs)
    print("tvecs : \n", tvecs)

    # 保存内参矩阵和畸变系数到文件
    # np.savez('calibration_data_right.npz', mtx=mtx, dist=dist)

    return {"mtx": mtx, "dist": dist, "rvecs": rvecs, "tvecs": tvecs}

   

def main():
    # 准备标定板上角点的物理坐标，假设标定板正好在世界坐标系的原点，
    # 例如，使用一个标准的棋盘格标定板，每个格子的大小为square_size，
    # 棋盘格的大小为 (width, height)。
    square_size = 2.0
    width, height = 11, 8
    image_folder = 'Hkvs_dataset/right'
    calibration_data = calibrate_camera(square_size, width, height, image_folder)


    # Save calibration data to file
    # **操作符：在Python中，**操作符用于将字典的键值对解包为关键字参数。这意味着如果你有一个字典，其键值对可以直接作为函数的关键字参数传递。
    np.savez('calibration_data_right.npz', **calibration_data)

if __name__ == "__main__":
    main()