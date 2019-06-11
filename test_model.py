'''
10. 验证单张图像
'''

image_path = '/train/train/cb7fb54008ef21a8b55da46d5145acb3.jpg'
img = Image.open(image_path)
img = ds_trans(img)#处理图像
#显示图像
inp = img.numpy().transpose((1, 2, 0))
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
inp = std * inp + mean
plt.imshow(inp)

model = model.cpu()
out = model(img)#获得输出
idx = torch.argmax(out).item()
cls = idx_to_class[idx]#获取测试图像类别
print('The breed of testing dog is: ',cls)
