import open3d as o3d
import numpy as np
import csv


def max_length_point(point, points):
    # 求最大距离对应的点
    l = 0
    max_point = points[0]
    for i in range(points.shape[0]):
        d = np.linalg.norm(point - points[i])
        if d > l:
            l = d
            max_point = points[i]
    return max_point


def min_length_point(point, points):
    # 求最小距离对应的点
    l = np.linalg.norm(point - points[0])
    min_point = points[0]
    for i in range(points.shape[0]):
        d = np.linalg.norm(point - points[i])
        if d < l:
            l = d
            min_point = points[i]
    return min_point


def point_to_line_distance(point, line_point1, line_point2):
    # 对于两点坐标为同一点时,返回点与点的距离
    if line_point1.all() == line_point2.all():
        point_array = np.array(point)
        point1_array = np.array(line_point1)
        return np.linalg.norm(point_array - point1_array)
    # 计算直线的三个参数
    A = line_point2[1] - line_point1[1]
    B = line_point1[0] - line_point2[0]
    C = (line_point1[1] - line_point2[1]) * line_point1[0] + \
        (line_point2[0] - line_point1[0]) * line_point1[1]
    # 根据点到直线的距离公式计算距离
    distance = np.abs(A * point[0] + B * point[1] + C) / (np.sqrt(A ** 2 + B ** 2))
    return distance


# 铜板长度计算
def tongban_length(name):
    file = "E:/python_file/anode_copper_ranging/res/" + name + ".txt"
    pcd = o3d.io.read_point_cloud(file, format='xyz')

    #将点云的坐标和颜色转换为numpy格式
    points = np.array(pcd.points)
    colors = np.array(pcd.colors)

    data = []
    with open(file, 'rt') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        for row in reader:
            data.append(row)
        #输出结果是列表
    data = np.array(data)
    data_ear = []
    data_body = []
    print("原始数据的shape：", data.shape)
    for i in range(data.shape[0]):
        if str(data[i, 3]) == "1.000000000000000000e+00":
            data_body.append([points[i, 0], points[i, 1], points[i, 2]])
        if str(data[i, 3]) == "2.000000000000000000e+00":
            data_ear.append([points[i, 0], points[i, 1], points[i, 2]])
    # 得到耳朵和板身的点
    data_ear = np.array(data_ear)
    data_body = np.array(data_body)
    print("耳朵的shape：", data_ear.shape, "板身的shape：", data_body.shape)
    # 计算耳朵与板身的平均点
    ear_average = np.average(data_ear, axis=0)
    # 求耳朵两端外侧的点
    ear_out1 = max_length_point(ear_average, data_ear)
    ear_out2 = max_length_point(ear_out1, data_ear)
    print("耳朵外侧两端的点：", ear_out1, ear_out2)
    out_ear_length = np.linalg.norm(ear_out1 - ear_out2)
    print("外耳宽：", out_ear_length)
    # 分开耳朵的点
    ear_data1 = []
    ear_data2 = []
    for i in range(data_ear.shape[0]):
        # 以外耳宽一半为阈值划分两个耳朵的点
        d = np.linalg.norm(ear_out1 - data_ear[i])
        if d < 0.5 * out_ear_length:
            ear_data1.append([data_ear[i, 0], data_ear[i, 1], data_ear[i, 2]])
        d = np.linalg.norm(ear_out2 - data_ear[i])
        if d < 0.5 * out_ear_length:
            ear_data2.append([data_ear[i, 0], data_ear[i, 1], data_ear[i, 2]])
    ear_data1 = np.array(ear_data1)
    ear_data2 = np.array(ear_data2)
    print("两个耳朵分开的shape：", ear_data1.shape, ear_data1.shape)
    # 求耳朵两端内侧的点
    ear_in1 = min_length_point(ear_out2, ear_data1)
    ear_in2 = min_length_point(ear_out1, ear_data2)
    print("耳朵内侧两端的点：", ear_in1, ear_in2)
    in_ear_length = np.linalg.norm(ear_in1 - ear_in2)
    print("内耳宽：", in_ear_length)
    # 求底部的两个点
    body_down1 = max_length_point(ear_out2, data_body)
    body_down2 = max_length_point(ear_out1, data_body)
    print("板身底部两端的点：", body_down1, body_down2)
    body_width = np.linalg.norm(body_down1 - body_down2)
    print("板宽：", body_width)
    # 求板身顶端边缘中间的点
    ear_out_average = (ear_out1 + ear_out2) * 0.5
    body_up = min_length_point(ear_out_average, data_body)
    print("板身顶端边缘的点：", body_up)
    body_down = (body_down1 + body_down2) * 0.5
    print("板身底端边缘的点：", body_down)
    body_up_height = point_to_line_distance(body_up, ear_out1, ear_out2)
    body_down_height = point_to_line_distance(body_down, ear_out1, ear_out2)
    body_height = body_down_height - body_up_height
    print("板高：", body_height)



    # 保存关键点便于观察
    points = []
    points.append(ear_out1)
    points.append(ear_out2)
    points.append(ear_in1)
    points.append(ear_in2)
    points.append(body_down1)
    points.append(body_down2)
    points.append(body_up)
    points.append(body_down)
    points = np.array(points)
    np.savetxt("E:/python_file/anode_copper_ranging/res/tongban_point.txt", points, delimiter=' ', fmt='%s')
    return out_ear_length, in_ear_length, body_width, body_height




if __name__ == "__main__":
    out_ear_length, in_ear_length, body_width, body_height = tongban_length("tongban")
    heigh = yuanzhu_length("yuanzhu")
