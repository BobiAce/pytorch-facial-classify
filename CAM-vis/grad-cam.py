# coding: utf-8
import cv2
import numpy as np
import torch
from torch.nn import functional as F
# import torchvision.transforms as transforms
from torch.autograd import Variable
import argparse
from PIL import Image
from torchvision import transforms
from models.resnetall import resnet34
from torchvision.models import vgg19
from models import VGG
import os

class GradCam:
	def __init__(self, model):
		self.model = model.eval()
		self.classes = {0: 'Surprise', 1: 'Fear', 2: 'Disgust', 3: 'Happy', 4: 'Sadness', 5: 'Angry', 6: 'Neutral'}
		self.feature = None
		self.gradient = None

	def save_gradient(self, grad):
		self.gradient = grad

	def __call__(self, x):
		# x = x.unsqueeze(0)#预测单张
		ncrops, c, h, w = np.shape(x)
		image_size = (x.size(-1), x.size(-2))
		datas = Variable(x)

		logit = self.model(datas)
		outputs_avg = logit.view(ncrops, -1).mean(0)  # avg over crops
		# h_x = F.softmax(outputs_avg, dim=1).data.squeeze()
		h_x = F.softmax(outputs_avg).data.squeeze()
		probs, idx = h_x.sort(0, True)
		probs = probs.numpy()
		idx = idx.numpy()
		for i in range(0, 5):
			print('{:.3f} -> {}'.format(probs[i], self.classes[idx[i]]))

		heat_maps = []
		for i in range(1):#datas.size(0)
			img = datas[i].data.cpu().numpy()
			img = img - np.min(img)
			if np.max(img) != 0:
				img = img / np.max(img)

			feature = datas[i].unsqueeze(0)
			LAYERNAME = self.model.named_children()
			for name, module in LAYERNAME:
				if name == 'fc':
					feature = feature.view(feature.size(0), -1)
				feature = module(feature)
				if name == 'conv5_x':
					feature.register_hook(self.save_gradient)
					self.feature = feature
			# classes = F.sigmoid(feature)
			classes = F.softmax(feature)
			one_hot, _ = classes.max(dim=-1)
			self.model.zero_grad()
			one_hot.backward()

			weight = self.gradient.mean(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
			mask = F.relu((weight * self.feature).sum(dim=1)).squeeze(0)
			mask = cv2.resize(mask.data.cpu().numpy(), image_size)
			mask = mask - np.min(mask)
			if np.max(mask) != 0:
				mask = mask / np.max(mask)
			heat_map = np.float32(cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET))
			cam = heat_map + np.float32((np.uint8(img.transpose((1, 2, 0)) * 255)))
			cam = cam - np.min(cam)
			if np.max(cam) != 0:
				cam = cam / np.max(cam)
			heat_maps.append(transforms.ToTensor()(cv2.cvtColor(np.uint8(255 * cam), cv2.COLOR_BGR2RGB)))
		heat_maps = torch.stack(heat_maps)
		return heat_maps,self.classes[idx[0]]


###load model
net = resnet34(num_classes=7)
print(net)
checkpoint = torch.load(os.path.join('../RAF_Resnet34-48', 'PublicTest_model.t7'),map_location='cpu')
net.load_state_dict(checkpoint['net'])
# net.eval()

cut_size = 44
# transform_test = transforms.Compose([
#     transforms.CenterCrop(cut_size),
#     transforms.ToTensor(),
# ])
transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])
IMAGE_PATH = 'test_CAM/'
SAVE_PATH = 'test_CAM/'
if __name__ == '__main__':
	# parser = argparse.ArgumentParser(description='Test Grad-CAM')
	# parser.add_argument('--image_name', default='test_CAM/test_1.jpg', type=str, help='the tested image name')
	# parser.add_argument('--save_name', default='images/cam_grad', type=str, help='saved image name')
	# opt = parser.parse_args()
	# IMAGE_NAME = opt.image_name
	# SAVE_NAME = opt.save_name

	imagelist = os.listdir(IMAGE_PATH)
	num = 0
	for image in imagelist:
		print(image)
		imgpath = IMAGE_PATH + image
		test_image = Image.open(imgpath).convert('RGB')

		img_tensor = transform_test(test_image)

		# crop_img = transforms.ToPILImage()(img_tensor)
		cv_image = cv2.cvtColor(np.asarray(test_image), cv2.COLOR_RGB2BGR)
		cv2.imwrite(SAVE_PATH + image, cv_image)
		cv2.imshow("OpenCV", cv_image)

		grad_cam = GradCam(net)
		feature_images,biaoqing = grad_cam(img_tensor)

		feature_image = feature_images.squeeze(dim=0)
		# feature_image = feature_images[0]
		feature_image = transforms.ToPILImage()(feature_image)

		heatmap_image = cv2.cvtColor(np.asarray(feature_image), cv2.COLOR_RGB2BGR)
		saveheatmap = os.path.join(SAVE_PATH,biaoqing + '-grad-cam_'+str(num) + '.jpg')
		cv2.imshow("heatmap",heatmap_image)
		cv2.imwrite(saveheatmap,heatmap_image)
		num = num + 1
		cv2.waitKey(0)
