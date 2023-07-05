import torch
import torchvision.models as models
import torch.nn as nn
import os
from PIL import Image
import cv2
import torchvision.transforms as transforms
import numpy as np

# 读取图像并进行大小调整和标准化处理
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

# 建立模型并修改全连接层
class_num = 525
model = models.__dict__['resnet18']()
model.fc = nn.Sequential(nn.Dropout(0.0), nn.Linear(model.fc.in_features,class_num))

# 模型并行处理并导入预训练CKPT
model = torch.nn.DataParallel(model).cuda()
CKPT = torch.load('logs/resnet18_oridata_epochs15_lr01_wd_1e-4_dropout00_optsgd/checkpoint-best.pth', map_location='cpu')['state_dict']
model.load_state_dict(CKPT)   #加载训练完保存好的模型

# 如果使用了 nn.DataParallel，它会自动打包模型并将其分布在多个 GPU 上，实现并行计算。在这种情况下，model 变量实际上指向 nn.DataParallel 对象，而不是模型本身。
# nn.DataParallel 对象的 .module 属性指向实际的模型。因此，如果要在 nn.DataParallel 模式下获取模型的属性（例如特征提取部分），需要通过 .module 属性获取实际的模型，然后对其进行操作
model_features = nn.Sequential(*list(model.module.children())[:-2])
fc_weights = model.state_dict()['module.fc.1.weight'].cpu().numpy()
model.eval()
model_features.eval()    

# 单张图片测试
img_path = 'data/test/AZARAS SPINETAIL/2.jpg'             
_, img_name = os.path.split(img_path)
img = Image.open(img_path).convert('RGB')
img_tensor = transform(img).unsqueeze(0)
img_tensor = img_tensor.to('cuda')

# 最终的特征图集合
features = model_features(img_tensor).detach().cpu().numpy()  
print(features.shape)
logit = model(img_tensor)
# 每个类别对应概率
h_x = torch.nn.functional.softmax(logit, dim=1).data.squeeze()  


# 打开数据文件
with open('data/birds.csv', "r") as f:
    content = f.readlines()

# 初始化一个空字典
class_to_scientific = {}

# 将每一行文本解析成字典中的一项
for line in content[1:]:
    # 去除行首行尾空格并用逗号分隔
    line = line.strip().split(',')
    # 将class id 和 scientific name 分别作为键和值存储到字典中
    class_to_scientific[int(line[0][:-2])] = line[4]

# 下面for循环的目的是依次打印该图片针对每个类别的概率
probs, idx = h_x.sort(0, True)      # 输出概率升序排列
probs = probs.cpu().numpy()  
idx = idx.cpu().numpy()  
for i in range(class_num):   
    print('{:.3f} -> {}'.format(probs[i], class_to_scientific[idx[i]]))  # 打印预测结果
print(img_name + ' output for the top1 prediction: %s' % class_to_scientific[idx[0]]) # y预测第一

def CAM(feature_conv, weight_softmax, class_idx):
    b, c, h, w = feature_conv.shape
    print(feature_conv.shape)
    
    output_cam = []
    for idx in class_idx:  #输出每个类别的预测效果
        cam = weight_softmax[idx].dot(feature_conv.reshape((c, h*w)))  
        print(weight_softmax[idx].shape)
        cam = cam.reshape(h, w)
        cam_img = (cam - cam.min()) / (cam.max() - cam.min())  #Normalize
        cam_img = np.uint8(255 * cam_img)  #Format as CV_8UC1 (as applyColorMap required)
        output_cam.append(cam_img)
    return output_cam
CAMs = CAM(features, fc_weights, [idx[0]])  #输出预测概率最大的特征图集对应的CAM

# 热力图和原图进行叠加
img = cv2.imread(img_path)
height, width, _ = img.shape  # 读取输入图片的尺寸
heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)  #CAM resize match input image size
result = heatmap * 0.3 + img * 0.5    # 比例可以自己调节

# 进行文字标识
text1 = '%s' % (class_to_scientific[idx[0]]) 	 #激活图结果上的文字显示
text2 = '%.2f%%' % (probs[0]*100) 				 #激活图结果上的文字显示
cv2.putText(result, text1, (5, 220), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
            color=(123, 222, 238), thickness=2, lineType=cv2.LINE_AA)
cv2.putText(result, text2, (5, 200), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6,
            color=(123, 222, 238), thickness=2, lineType=cv2.LINE_AA)

# 存储CAM结果
CAM_RESULT_PATH = r'imgs/'   
if not os.path.exists(CAM_RESULT_PATH):
    os.mkdir(CAM_RESULT_PATH)
image_name_ = img_name.split(".")[-2]
cv2.imwrite(CAM_RESULT_PATH + image_name_ + '_' + 'pred_' + class_to_scientific[idx[0]] + '.jpg', result)  #写入存储磁盘

