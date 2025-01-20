import logging
import numpy as np
import torch
import sys

from HAIAFusion.models.fusion_model import HAIAFusion
from MBHFuse.models.Fusion import MBHFuse
from PIL import Image
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QPushButton, QLabel, QWidget, QFileDialog, QHBoxLayout, QTabWidget


# 配置日志
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


def clamp(image, min_val=0, max_val=1):
    return torch.clamp(image, min_val, max_val)


def YCrCb2RGB(y, cb, cr):
    ycbcr = np.stack([y, cb, cr], axis=-1)
    rgb = Image.fromarray(ycbcr, 'YCbCr').convert('RGB')
    return rgb


class ImageFusionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("适应特殊环境的红外与可见光图像融合")
        self.setGeometry(100, 100, 1000, 800)

        # 加载样式表
        self.load_stylesheet()

        # 检查是否有可用的GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化模型
        self.haia_fusion_model = None
        self.mbh_fuse_model = None
        self.load_fusion_models()

        # 图像路径和展示标签
        self.infrared_image_path_tab1 = None
        self.visible_image_path_tab1 = None
        self.infrared_image_path_tab2 = None
        self.visible_image_path_tab2 = None

        # 创建界面
        self.initUI()

    def load_stylesheet(self):
        try:
            with open('style.qss', 'r', encoding='utf-8') as f:
                self.setStyleSheet(f.read())
        except FileNotFoundError:
            logging.error("样式表文件 style.qss 未找到")
        except Exception as e:
            logging.error(f"加载样式表失败: {e}")

    def initUI(self):
        # 创建TabWidget
        tab_widget = QTabWidget()
        self.tab1 = QWidget()
        self.tab2 = QWidget()

        tab_widget.addTab(self.tab1, "低光环境融合")
        tab_widget.addTab(self.tab2, "复杂环境融合")

        # Tab1 - 低光环境融合
        main_layout_tab1 = QVBoxLayout(self.tab1)

        # 水平布局用于展示图像
        image_layout_tab1 = QHBoxLayout()
        self.infrared_label_tab1 = QLabel("红外图像展示", self)
        self.visible_label_tab1 = QLabel("可见光图像展示", self)
        self.fused_label_tab1 = QLabel("融合结果展示", self)
        image_layout_tab1.addWidget(self.infrared_label_tab1)
        image_layout_tab1.addWidget(self.visible_label_tab1)
        image_layout_tab1.addWidget(self.fused_label_tab1)
        main_layout_tab1.addLayout(image_layout_tab1)

        # 水平布局用于放置按钮
        button_layout_tab1 = QHBoxLayout()

        # 导入红外图像按钮
        import_infrared_button_tab1 = QPushButton("导入红外图像", self)
        import_infrared_button_tab1.clicked.connect(lambda: self.import_image("infrared", self.tab1))
        button_layout_tab1.addWidget(import_infrared_button_tab1)

        # 导入可见光图像按钮
        import_visible_button_tab1 = QPushButton("导入可见光图像", self)
        import_visible_button_tab1.clicked.connect(lambda: self.import_image("visible", self.tab1))
        button_layout_tab1.addWidget(import_visible_button_tab1)

        # 低光环境融合按钮
        low_light_button = QPushButton("低光环境红外与可见光图像融合模块", self)
        low_light_button.clicked.connect(lambda: self.run_fusion("low_light"))
        button_layout_tab1.addWidget(low_light_button)

        # 将按钮布局添加到主布局中
        main_layout_tab1.addLayout(button_layout_tab1)

        # 状态标签
        self.status_label_tab1 = QLabel("请选择图像进行融合", self)
        self.status_label_tab1.setAlignment(Qt.AlignCenter)
        main_layout_tab1.addWidget(self.status_label_tab1)

        # Tab2 - 复杂环境融合
        main_layout_tab2 = QVBoxLayout(self.tab2)

        # 水平布局用于展示图像
        image_layout_tab2 = QHBoxLayout()
        self.infrared_label_tab2 = QLabel("红外图像展示", self)
        self.visible_label_tab2 = QLabel("可见光图像展示", self)
        self.fused_label_tab2 = QLabel("融合结果展示", self)
        image_layout_tab2.addWidget(self.infrared_label_tab2)
        image_layout_tab2.addWidget(self.visible_label_tab2)
        image_layout_tab2.addWidget(self.fused_label_tab2)
        main_layout_tab2.addLayout(image_layout_tab2)

        # 水平布局用于放置按钮
        button_layout_tab2 = QHBoxLayout()

        # 导入红外图像按钮
        import_infrared_button_tab2 = QPushButton("导入红外图像", self)
        import_infrared_button_tab2.clicked.connect(lambda: self.import_image("infrared", self.tab2))
        button_layout_tab2.addWidget(import_infrared_button_tab2)

        # 导入可见光图像按钮
        import_visible_button_tab2 = QPushButton("导入可见光图像", self)
        import_visible_button_tab2.clicked.connect(lambda: self.import_image("visible", self.tab2))
        button_layout_tab2.addWidget(import_visible_button_tab2)

        # 复杂环境融合按钮
        complex_button = QPushButton("复杂环境红外与可见光图像融合模块", self)
        complex_button.clicked.connect(lambda: self.run_fusion("complex"))
        button_layout_tab2.addWidget(complex_button)

        # 将按钮布局添加到主布局中
        main_layout_tab2.addLayout(button_layout_tab2)

        # 状态标签
        self.status_label_tab2 = QLabel("请选择图像进行融合", self)
        self.status_label_tab2.setAlignment(Qt.AlignCenter)
        main_layout_tab2.addWidget(self.status_label_tab2)

        # 设置中央窗口
        container = QWidget()
        container.setLayout(QVBoxLayout())
        container.layout().addWidget(tab_widget)
        self.setCentralWidget(container)

    def load_fusion_models(self):
        try:
            haia_model_path = 'HAIAFusion/pretrained/fusion_model_epoch_99.pth'
            mbh_model_path = 'MBHFuse/pretrained/fusion_model_epoch_59.pth'

            self.haia_fusion_model = HAIAFusion().to(self.device)
            self.haia_fusion_model.load_state_dict(torch.load(haia_model_path, map_location=self.device))
            self.haia_fusion_model.eval()  # 切换到评估模式

            self.mbh_fuse_model = MBHFuse().to(self.device)
            pretrained_dict = torch.load(mbh_model_path, map_location=self.device)
            model_dict = self.mbh_fuse_model.state_dict()

            # 过滤出预训练模型中与模型定义匹配的层，并移除 module. 前缀
            pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items() if k.replace('module.', '') in model_dict}
            model_dict.update(pretrained_dict)
            self.mbh_fuse_model.load_state_dict(model_dict)

            self.mbh_fuse_model.eval()  # 切换到评估模式

            print("模型加载成功")
        except Exception as e:
            logging.error(f"加载模型失败: {e}")

    def import_image(self, image_type, tab):
        try:
            if tab == self.tab1:
                if image_type == "infrared":
                    self.infrared_image_path_tab1, _ = QFileDialog.getOpenFileName(self, "选择红外图像", "", "Images (*.png *.jpg *.bmp)")
                    if self.infrared_image_path_tab1:
                        self.status_label_tab1.setText(f"已导入红外图像: {self.infrared_image_path_tab1}")
                        self.display_image(self.infrared_image_path_tab1, self.infrared_label_tab1)
                elif image_type == "visible":
                    self.visible_image_path_tab1, _ = QFileDialog.getOpenFileName(self, "选择可见光图像", "", "Images (*.png *.jpg *.bmp)")
                    if self.visible_image_path_tab1:
                        self.status_label_tab1.setText(f"已导入可见光图像: {self.visible_image_path_tab1}")
                        self.display_image(self.visible_image_path_tab1, self.visible_label_tab1)
            elif tab == self.tab2:
                if image_type == "infrared":
                    self.infrared_image_path_tab2, _ = QFileDialog.getOpenFileName(self, "选择红外图像", "", "Images (*.png *.jpg *.bmp)")
                    if self.infrared_image_path_tab2:
                        self.status_label_tab2.setText(f"已导入红外图像: {self.infrared_image_path_tab2}")
                        self.display_image(self.infrared_image_path_tab2, self.infrared_label_tab2)
                elif image_type == "visible":
                    self.visible_image_path_tab2, _ = QFileDialog.getOpenFileName(self, "选择可见光图像", "", "Images (*.png *.jpg *.bmp)")
                    if self.visible_image_path_tab2:
                        self.status_label_tab2.setText(f"已导入可见光图像: {self.visible_image_path_tab2}")
                        self.display_image(self.visible_image_path_tab2, self.visible_label_tab2)
        except Exception as e:
            logging.error(f"导入图像失败: {e}")

    def display_image(self, image_path, label):
        try:
            image = Image.open(image_path)
            display_size = (600, 600)

            # Use LANCZOS for high-quality downscaling
            image = image.resize(display_size, Image.Resampling.LANCZOS)

            # Ensure the image is in RGB mode
            if image.mode != 'RGB':
                image = image.convert("RGB")

            # Convert to QImage
            data = image.tobytes("raw", "RGB")
            qimage = QImage(data, image.width, image.height, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage)
            label.setPixmap(pixmap)
            label.setAlignment(Qt.AlignCenter)
        except Exception as e:
            logging.error(f"显示图像失败: {e}")

    def run_fusion(self, fusion_type):
        try:
            if fusion_type == "low_light":
                if self.haia_fusion_model is None:
                    self.status_label_tab1.setText("低光环境融合模型加载失败")
                    return

                if not self.infrared_image_path_tab1 or not self.visible_image_path_tab1:
                    self.status_label_tab1.setText("请先导入红外图像和可见光图像")
                    return

                fused_image = self.run_fusion_model(self.haia_fusion_model, self.infrared_image_path_tab1, self.visible_image_path_tab1)
                if fused_image is not None:
                    self.status_label_tab1.setText("低光环境红外与可见光图像融合完成")
                    self.display_fused_image(fused_image, self.fused_label_tab1)
                else:
                    self.status_label_tab1.setText("图像融合失败")
            elif fusion_type == "complex":
                if self.mbh_fuse_model is None:
                    self.status_label_tab2.setText("复杂环境融合模型加载失败")
                    return

                if not self.infrared_image_path_tab2 or not self.visible_image_path_tab2:
                    self.status_label_tab2.setText("请先导入红外图像和可见光图像")
                    return

                fused_image = self.run_fusion_model(self.mbh_fuse_model, self.infrared_image_path_tab2, self.visible_image_path_tab2)
                if fused_image is not None:
                    self.status_label_tab2.setText("复杂环境红外与可见光图像融合完成")
                    self.display_fused_image(fused_image, self.fused_label_tab2)
                else:
                    self.status_label_tab2.setText("图像融合失败")
        except Exception as e:
            logging.error(f"运行融合失败: {e}")


    def run_fusion_model(self, model, infrared_path, visible_path):
        try:
            # 加载可见光和红外图像
            infrared_image = Image.open(infrared_path).convert('L')
            visible_image = Image.open(visible_path).convert('RGB')

            # 将可见光图像转换为 YCrCb 颜色空间
            visible_ycbcr = visible_image.convert('YCbCr')
            y, cb, cr = visible_ycbcr.split()

            # 调整红外图像的大小与 Y 通道匹配
            if infrared_image.size != y.size:
                infrared_image = infrared_image.resize(y.size, Image.BILINEAR)

            # 转换为张量并归一化
            infrared_tensor = torch.tensor(np.array(infrared_image) / 255.0).unsqueeze(0).unsqueeze(0).float().to(
                self.device)
            y_tensor = torch.tensor(np.array(y) / 255.0).unsqueeze(0).unsqueeze(0).float().to(self.device)

            # 进行融合
            with torch.no_grad():
                fused_tensor = model(y_tensor, infrared_tensor)

            # 将融合结果限制在 [0, 1] 范围内
            fused_tensor = clamp(fused_tensor)

            # 转换为图像格式并重建彩色图像
            fused_y = fused_tensor.squeeze().cpu().numpy() * 255.0
            rgb_fused_image = YCrCb2RGB(fused_y.astype(np.uint8), np.array(cb), np.array(cr))

            return rgb_fused_image
        except Exception as e:
            logging.error(f"图像融合失败: {e}")
            return None

    def display_fused_image(self, fused_image, label):
        try:
            # 调整图像大小并转换为 QImage
            image = fused_image.resize((600, 600))
            qimage = QImage(image.tobytes(), image.width, image.height, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage)
            label.setPixmap(pixmap)
            label.setAlignment(Qt.AlignCenter)
        except Exception as e:
            logging.error(f"显示融合图像失败: {e}")



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageFusionApp()
    window.show()
    sys.exit(app.exec_())