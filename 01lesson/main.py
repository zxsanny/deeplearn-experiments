from torchvision import models
from torchvision import transforms
from PIL import Image
import torch.nn.functional


preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

img = Image.open('img/car.jpg')
img_tensor = preprocess(img)

resnet = models.resnet101(pretrained=True)
resnet.eval()
out = resnet(torch.unsqueeze(img_tensor, 0))

with open('imagenet_classes.txt') as f:
    labels = [line.strip() for line in f.readlines()]

percentage = torch.nn.functional.softmax(out, dim=1)[0]*100
_, indices = torch.sort(out, descending=True)
result = [(labels[idx], percentage[idx].item()) for idx in indices[0][:7]]
print(result)
