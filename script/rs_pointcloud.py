import pyrealsense2 as rs
import numpy as np
import open3d as o3d

# 配置深度和彩色流
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 开始流
pipeline.start(config)

try:
    # 等待一个连贯的帧：深度和彩色
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not depth_frame or not color_frame:
        raise RuntimeError("Could not acquire depth or color frames.")

    # 将图像转换为numpy数组
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # 创建点云
    pc = rs.pointcloud()
    pc.map_to(color_frame)
    points = pc.calculate(depth_frame)
    vtx = np.asanyarray(points.get_vertices())

    # 从vtx中提取x, y, z坐标
    xyz = np.array([[v[0], v[1], v[2]] for v in vtx], dtype=np.float64)
    
    # 使用Open3D将点云转换为Open3D的点云格式
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    
    # 可以选择为点云上色
    color = np.asanyarray(color_frame.get_data())
    color = color.reshape((-1, 3))
    color = color[:, [2, 1, 0]] / 255.0  # BGR到RGB，归一化颜色值
    pcd.colors = o3d.utility.Vector3dVector(color)

    # 保存点云
    o3d.io.write_point_cloud("output.ply", pcd)

    print("点云已保存为 output.ply")

      # 显示点云
    o3d.visualization.draw_geometries([pcd], window_name="RealSense Point Cloud")

finally:
    pipeline.stop()
