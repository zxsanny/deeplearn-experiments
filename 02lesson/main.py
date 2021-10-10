import torch
from resgen import ResNetGenerator
from PIL import Image
from torchvision import transforms

# prepare image to tensor
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor()
])
img = Image.open('horse01.jpg')
img_t = preprocess(img)
batch_t = torch.unsqueeze(img_t, 0)

# NN
model_data = torch.load('../data/02/horse2zebra_0.4.0.pth')
netG = ResNetGenerator()
netG.load_state_dict(model_data)
netG.eval()
batch_out = netG(batch_t)

# tensor to image
out_t = (batch_out.data.squeeze() + 1.0) / 2.0
out_img = transforms.ToPILImage()(out_t)
out_img.save('zebra01.jpg')
