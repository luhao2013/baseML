import numpy as np

# 可以参考 https://zhuanlan.zhihu.com/p/63974249 步骤更细
def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """
    将输入数据转化为适合做卷积运算的二维形式
    :param input_data:由(batch, channel, height, width)的4维数组构成的输入数据
    :param filter_h:滤波器的高
    :param filter_w:滤波器的宽
    :param stride:步长
    :param pad:填充
    :return:二维数组
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h) // stride + 1
    out_w = (W + 2*pad - filter_w) // stride + 1

    # N C两维不需要填充，故为0，最后两维填充
    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)

    return col
