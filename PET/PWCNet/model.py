import torch
import PIL.Image
import numpy as np

class OpticalFlowModel:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(model_path)
    
    def load_model(self, model_path):
        # 加载模型
        model = torch.load(model_path, map_location=self.device)
        model.eval()
        return model
    
    def preprocess_image(self, image):
        # 预处理图像
        # image = PIL.Image.open(image_path)
        image = image.resize((512, 512))
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0)
        return image.to(self.device)
    
    def compute_flow(self, image1, image2):
        # 计算光流
        tenOne = self.preprocess_image(image1)
        tenTwo = self.preprocess_image(image2)
        
        with torch.no_grad():
            flow = self.model(tenOne, tenTwo)
        return flow.cpu().detach().numpy()

# 创建光流模型的函数
def create_optical_flow_model(model_path):
    return OpticalFlowModel(model_path)
