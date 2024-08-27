# !/usr/bin/env python
# -*-coding:utf-8 -*-

from PIL import Image
import onnxruntime
import numpy as np
from io import BytesIO

np.set_printoptions(precision=4)


class PreONNX(object):
    def __init__(self, path, providers=None):
        # 初始化 ONNX 模型
        if not providers:
            providers = ['CPUExecutionProvider']  # 默认使用 CPU 执行提供者
        self.sess = onnxruntime.InferenceSession(path, providers=providers)
        self.loadSize = 512  # 加载图像的默认大小
        self.input_shape = [105, 105]  # 输入图像的形状

    def sigmoid(self, x):
        # Sigmoid 激活函数
        return 1 / (1 + np.exp(-x))

    def convert_to_image(self, file):
        """
        将输入文件转换为 PIL 图像对象
        :param file: 可以是 ndarray、字节流、PIL 图像对象或文件路径
        :return: PIL 图像对象
        """
        # 图片转换为矩阵
        if isinstance(file, np.ndarray):
            img = Image.fromarray(file)  # 如果输入是 ndarray，转换为 PIL 图像
        elif isinstance(file, bytes):
            img = Image.open(BytesIO(file))  # 如果输入是字节流，转换为 PIL 图像
        elif isinstance(file, Image.Image):
            img = file  # 如果输入已经是 PIL 图像，直接使用
        else:
            img = Image.open(file)  # 其他情况，假设输入是文件路径，打开图像
        return img

    def load_and_process_image(self, file, input_shape, nc=3):
        """
        打开并处理图像
        :param file: 可以是 ndarray、字节流、PIL 图像对象或文件路径
        :param input_shape: 目标图像的尺寸 (height, width)
        :param nc: 通道数，默认为 3（RGB），如果为 1 则转换为灰度图像
        :return: 处理后的 PIL 图像对象
        """
        # 打开并处理图像
        img = self.convert_to_image(file)
        img = img.convert('RGB')  # 转换为 RGB 模式
        h, w = input_shape
        img = img.resize((w, h), Image.Resampling.LANCZOS)  # 调整图像大小，使用高质量的 LANCZOS 采样
        if nc == 1:
            img = img.convert('L')  # 如果需要单通道，转换为灰度图像
        return img

    def set_img(self, lines):
        """
        预处理图像，将其转换为模型输入所需的格式
        :param lines: 输入图像，可以是 ndarray、字节流、PIL 图像对象或文件路径
        :return: 预处理后的图像，格式为 (1, 3, height, width)
        """
        # 预处理图像
        image = self.load_and_process_image(lines, self.input_shape, nc=3)
        image = np.array(image).astype(np.float32) / 255.0  # 归一化
        photo = np.expand_dims(np.transpose(image, (2, 0, 1)), 0)  # 调整维度顺序并扩展维度
        return photo

    def reason(self, image_1, image_2):
        """
        对一对图像进行推理，返回模型的输出
        :param image_1: 第一张输入图像，可以是 ndarray、字节流、PIL 图像对象或文件路径
        :param image_2: 第二张输入图像，可以是 ndarray、字节流、PIL 图像对象或文件路径
        :return: 模型的输出，经过 Sigmoid 激活函数处理后的值
        """
        # 预处理图像
        photo_1 = self.set_img(image_1)
        photo_2 = self.set_img(image_2)

        # 运行模型
        out = self.sess.run(None, {"x1": photo_1, "x2": photo_2})

        # 处理模型输出
        out = out[0]
        out = self.sigmoid(out)
        out = out[0][0]

        return out

    def reason_all(self, image_1, image_2_list):
        """
        对多对图像进行推理，返回模型的输出列表
        :param image_1: 第一张输入图像，可以是 ndarray、字节流、PIL 图像对象或文件路径
        :param image_2_list: 第二张输入图像的列表，每个元素可以是 ndarray、字节流、PIL 图像对象或文件路径
        :return: 模型的输出列表，经过 Sigmoid 激活函数处理后的值
        """
        # 预处理第一张图像
        photo_1 = self.set_img(image_1)

        # 初始化 photo_2_all 和 photo_1_all
        photo_2_all = []
        photo_1_all = []

        # 预处理 image_2_list 中的每张图像，并将其与 photo_1 连接
        for image_2 in image_2_list:
            photo_2 = self.set_img(image_2)
            photo_2_all.append(photo_2)
            photo_1_all.append(photo_1)

        # 将列表转换为 numpy 数组，并沿第一个维度连接
        photo_2_all = np.concatenate(photo_2_all, axis=0)
        photo_1_all = np.concatenate(photo_1_all, axis=0)

        # 运行模型
        out = self.sess.run(None, {"x1": photo_1_all, "x2": photo_2_all})

        # 处理模型输出
        out = out[0]
        out = self.sigmoid(out)
        out = out.tolist()
        out = [i[0] for i in out]  # 提取每个输出的第一个元素

        return out


if __name__ == '__main__':
    pre_onnx_path = "pre_model.onnx"
    pre = PreONNX(pre_onnx_path, providers=['CPUExecutionProvider'])
    image_1 = r"datasets\bilbil\character2\char_1.jpg"
    image_2 = r"datasets\bilbil\character2\plan_4.jpg"
    image_1 = "img.png"
    image_2 = "img_1.png"
    large_img = pre.reason_all(image_1, image_2)
    print(large_img)
