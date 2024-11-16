import torch
from PIL import Image
import cv2


def main():
    print(torch.torch_version)
    print(torch.cuda.is_available())


if __name__ == '__main__':
    main()
