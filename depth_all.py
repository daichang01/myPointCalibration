import cv2
import numpy as np
import open3d as o3d
from binocular_correction import rectify_images  # 确保这个模块正确实现了

def load_maps(filename):
    """加载立体校正映射数据。"""
    data = np.load(filename)
    return data['map1x'], data['map1y'], data['map2x'], data['map2y']

def compute_disparity(left_img_path, right_img_path, calibration_file, num_disparities, block_size):
    """从校正后的立体图像中计算视差图。"""
    map1x, map1y, map2x, map2y = load_maps(calibration_file)
    rectified_left, rectified_right = rectify_images(left_img_path, right_img_path, map1x, map1y, map2x, map2y)

    stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=num_disparities, blockSize=block_size)
    disparity = stereo.compute(rectified_left, rectified_right).astype(np.float32) / 16.0
    return disparity

def save_point_cloud(disparity, output_file, baseline, focal_length, cx, cy):
    """从视差图生成并保存点云。"""
    points = []
    height, width = disparity.shape
    for y in range(height):
        for x in range(width):
            if disparity[y][x] != 0 and disparity[y][x] != -16 :
                Z = (focal_length * baseline) / disparity[y, x]
                X = (x - cx) * Z / focal_length
                Y = (y - cy) * Z / focal_length
                points.append([X, Y, Z])
    points = np.array(points)
    create_output_file(points, output_file)

def create_output_file(vertices, filename):
    """从顶点数据创建PLY文件。"""
    ply_header = '''ply
format ascii 1.0
element vertex {}
property float x
property float y
property float z
end_header
'''
    with open(filename, 'w') as f:
        f.write(ply_header.format(len(vertices)))
        np.savetxt(f, vertices, '%f %f %f')

def main():
    left_img_path = "teeth_left.bmp"
    right_img_path = "teeth_right.bmp"
    calibration_file = "calibration_maps.npz"
    output_file = "output_point_cloud.ply"
    baseline = 41.9  # 相机之间的基线距离
    focal_length = 4745  # 相机的焦距
    cx, cy = 1250, 1210  # 主点

    disparity = compute_disparity(left_img_path, right_img_path, calibration_file, 16*2, 5)
    save_point_cloud(disparity, output_file, baseline, focal_length, cx, cy)

    # 加载并可视化生成的点云
    pcd = o3d.io.read_point_cloud(output_file)
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    main()
