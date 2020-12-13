import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T

transform_img = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def imShow(path):
  image = cv2.imread(path)
  height, width = image.shape[:2]
  resized_image = cv2.resize(image,(3*width, 3*height), interpolation = cv2.INTER_CUBIC)

  fig = plt.gcf()
  fig.set_size_inches(18, 10)
  plt.axis("off")
  plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
  plt.show()


def imshow(input_tensor, title=None):
    """Imshow for Tensor."""
    input_np = input_tensor.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    input_np = std * input_np + mean
    input_np = np.clip(input_np, 0, 1)
    plt.imshow(input_np)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


def infer_image_with_EN(model, image, device, class_names_list):
    """
    model: EfficentNet Model.
    image: PIL image.
    """
    was_training = model.training
    model.eval()

    num_cnt = 0
    transform_img = T.Compose([T.Resize((224, 224)),
                               T.ToTensor(),
                               T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    inputimg = transform_img(image).unsqueeze(0)

    with torch.no_grad():
        inputs = inputimg.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, axis=1)
        inference = class_names_list[int(preds.view(-1).cpu())]

        # print(f'test done - inference : {inference}')

    model.train(mode=was_training)
    return inputimg, inference