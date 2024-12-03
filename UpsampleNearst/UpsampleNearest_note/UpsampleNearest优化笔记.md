1. UpsampleNearst
		数学表达式：
		对于输出图像的每个像素 (x_out, y_out)，对应输入图像的像素 (x_in, y_in) 计算方式为：
		x_in = round(x_out * input_width / output_width)
		y_in = round(y_out * input_height / output_height)
		然后，输出图像的像素值为输入图像 (x_in, y_in) 处的像素值。