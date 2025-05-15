import sys
import open3d as o3d

if len(sys.argv) != 2:
    print("Usage: python view_pcd.py <pointcloud.ply>")
    sys.exit(1)

pcd = o3d.io.read_point_cloud(sys.argv[1])
o3d.visualization.draw_geometries([pcd])
