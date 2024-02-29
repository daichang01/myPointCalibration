import cv2 
import numpy as np
import open3d as o3d


def load_maps(filename):
    """加载立体校正映射数据。"""
    data = np.load(filename)
    return data['map1x'], data['map1y'], data['map2x'], data['map2y']


img_left = cv2.imread("rectified_left.bmp")
img_right = cv2.imread("rectified_right.bmp")

map1x, map1y, map2x, map2y = load_maps("calibration_maps.npz")  

img_left_rectified = cv2.remap(img_left, map1x, map1y, cv2.INTER_LINEAR)
img_right_rectified = cv2.remap(img_right, map2x, map2y,cv2.INTER_LINEAR)
img_color = cv2.cvtColor(img_left_rectified, cv2.COLOR_BGR2RGB)
print(img_color.shape)

imgL = cv2.cvtColor(img_left_rectified, cv2.COLOR_BGR2GRAY)
imgR = cv2.cvtColor(img_right_rectified, cv2.COLOR_BGR2GRAY)

num = 10   # numDisparities（视差范围数量）
blockSize = 20 # blockSize（块大小）

#SGBM算法算出的视差值会乘以16
stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=16 * num, blockSize=blockSize)
disparity = stereo.compute(imgL, imgR)

dis_color = disparity
dis_color = cv2.normalize(dis_color, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype = cv2.CV_8U)
dis_color = cv2.applyColorMap(dis_color, 2)
cv2.imshow("depth", dis_color)
cv2.waitKey(0)
cv2.destroyWindow("depth")

b= 9  # 相机之间的基线距离
f = 4743 # 相机的焦距
cx, cy = 1319, 1033   # 主点

i = 0
output_points = np.zeros((2448 * 2048, 6))
for row in range(disparity.shape[0] - 1):
    for col in range(disparity.shape[1] - 1):
        dis = disparity[row][col]
        if dis != 0 and dis != (-16) and dis > 2000 and dis < 3000:
            if i < len(output_points):
                output_points[i][0] = 16*b*(col-cx)/dis
                output_points[i][1] = 16*b*(row-cy)/dis
                output_points[i][2] = 16*b*f/dis
                output_points[i][3] = img_color[row][col][0]
                output_points[i][4] = img_color[row][col][1]
                output_points[i][5] = img_color[row][col][2]
                i += 1
            else:
                print(f"Warning: Trying to access index {i}, which is out of bounds.")
    # 根据需要处理数组越界的情况，例如通过中断循环或其他方式
output_points = output_points[:i]  # 仅保存非零点


def create_output(vertices, filename):
    ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''
    with open(filename,'w') as file:
        file.write(ply_header%dict(vert_num = len(vertices)))
        np.savetxt(file, vertices, '%f %f %f %d %d %d')

output_file = 'dc.ply'
create_output(output_points, output_file)
pcd = o3d.io.read_point_cloud(output_file)
o3d.visualization.draw_geometries([pcd])
cv2.waitKey(0)


 ### ######               

# with open('point_cloud2.txt', 'w') as file:
#     for point in output_points[:i]:  # 假设 i 是有效数据的数量
#         file.write(f"{point[0]} {point[1]} {point[2]} {point[3]} {point[4]} {point[5]}\n")
# try:
#     points = np.loadtxt('point_cloud2.txt')
#     if points.size > 0:
#         # 创建 Open3D 点云对象
#         pcd = o3d.geometry.PointCloud()
#         # 设置点云的坐标
#         pcd.points = o3d.utility.Vector3dVector(points[:, :3])
#         # 设置点云的颜色
#         pcd.colors = o3d.utility.Vector3dVector(points[:, 3:6] / 255.0)
#         # 可视化点云
#         o3d.visualization.draw_geometries([pcd])
#     else:
#         print("点云文件为空，无法加载点云。")
# except Exception as e:
#     print(f"加载点云文件时出错: {e}")
