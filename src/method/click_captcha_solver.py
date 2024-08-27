import os

from src.utils import utils
from src.utils.ver_onnx import PreONNX
from src.utils.yolo_onnx import YOLOV5_ONNX


class ClickCaptchaSolver(object):
    def __init__(self, per_path='pre_model_v3.bin', yolo_path='best_v2.bin', sign=True):
        """
        jiyan 最好 pre_model_v3.onnx
        nine 最好  pre_model_v5.onnx
        """
        save_path = os.path.join(os.path.dirname(__file__), '../../model')
        path = lambda a, b: os.path.join(a, b)
        per_path = path(save_path, per_path)
        yolo_path = path(save_path, yolo_path)
        if sign:
            try:
                from src.utils.load import decryption
            except:
                raise Exception("Error! 请在windows下的python3.6、3.8、3.10环境下使用")
            yolo_path = decryption(yolo_path)
            per_path = decryption(per_path)
        self.yolo = YOLOV5_ONNX(yolo_path, classes=['target', 'title', 'char'], providers=['CPUExecutionProvider'])
        self.pre = PreONNX(per_path, providers=['CPUExecutionProvider'])

    def run(self, image_path):
        """
        检测
        :param img: 图片的路径、二进制数据或图片矩阵
        :return: list ---> [{'crop': [x1, y1, x2, y2], 'classes': ''}
        """
        img = utils.open_image(image_path)  # 打开图像
        data = self.yolo.decect(image_path)  # 使用 YOLO 检测图像中的目标
        # 需要选择的字
        targets = [i.get("crop") for i in data if i.get("classes") == "target"]  # 获取目标区域的坐标
        chars = [i.get("crop") for i in data if i.get("classes") == "char"]  # 获取字符区域的坐标
        # 根据坐标进行排序
        chars.sort(key=lambda x: x[0])  # 按照字符的 x 坐标进行排序
        print('chars: ', chars)
        chars = [img.crop(char) for char in chars]  # 裁剪出字符区域的图像
        result = []
        for m, img_char in enumerate(chars):  # 遍历每个字符图像
            if len(targets) == 0:
                break  # 如果没有目标区域，退出循环
            elif len(targets) == 1:
                slys_index = 0  # 如果只有一个目标区域，直接选择它
            else:
                img_target_list = []
                for n, target in enumerate(targets):  # 遍历每个目标区域
                    img_target = img.crop(target)  # 裁剪出目标区域的图像
                    img_target_list.append(img_target)  # 将目标区域图像添加到列表中
                # slys 相似度
                slys = self.pre.reason_all(img_char, img_target_list)  # 对字符图像和所有目标图像进行推理
                slys_index = slys.index(max(slys))  # 找到推理结果中最大值的索引
            result.append(targets[slys_index])  # 将对应的目标区域添加到结果中
            targets.pop(slys_index)  # 从目标列表中移除已经匹配的目标区域
            if len(targets) == 0:
                break  # 如果没有剩余的目标区域，退出循环
        return result  # 返回结果列表


if __name__ == '__main__':
    image_path = "../../docs/res.jpg"
    cap = ClickCaptchaSolver()
    result = cap.run(image_path)
    print(result)
    utils.drow_img(image_path, result, "click_captcha_solver.jpg")



