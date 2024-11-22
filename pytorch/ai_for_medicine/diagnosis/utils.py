import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from sklearn.metrics import roc_auc_score, roc_curve

random.seed(a=None, version=2)


def get_mean_std_per_batch(image_path, df, H=320, W=320):
    sample_data = []
    transform = transforms.Compose([
        transforms.Resize((H, W)),
        transforms.ToTensor()
    ])
    for img in df.sample(100)["Image"].values:
        sample_data.append(
            transform(cv2.imread(image_path + img)).numpy()
        )

    mean = np.mean(sample_data[0])
    std = np.std(sample_data[0])
    return mean, std


def load_image(img, image_dir, df, preprocess=True, H=320, W=320):
    """Load and preprocess image."""
    img_path = image_dir + img
    mean, std = get_mean_std_per_batch(img_path, df, H=H, W=W)
    img_array = cv2.imread(img_path)
    transform = transforms.Compose([
        transforms.Resize((H, W)),
        transforms.ToTensor()
    ])
    x = transform(img_array)
    if preprocess:
        x = (x - mean) / std
        x = x.unsqueeze(0)
    return x


def grad_cam(model, image, cls, layer_name, H=320, W=320):
    """GradCAM method for visualizing input saliency."""
    model.eval()

    def forward_hook(module, input, output):
        model.feature_maps = output

    def backward_hook(module, grad_in, grad_out):
        model.gradients = grad_out[0]

    # Hooking the specified layer
    for name, module in model.named_modules():
        if name == layer_name:
            module.register_forward_hook(forward_hook)
            module.register_backward_hook(backward_hook)

    output = model(image)
    one_hot_output = torch.zeros_like(output)
    one_hot_output[0, cls] = 1
    output.backward(gradient=one_hot_output)

    feature_maps = model.feature_maps[0]
    gradients = model.gradients[0]

    weights = gradients.mean(dim=(1, 2), keepdim=True)
    cam = (weights * feature_maps).sum(dim=0).detach().numpy()

    # Process CAM
    cam = cv2.resize(cam, (W, H), interpolation=cv2.INTER_LINEAR)
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    return cam


def compute_gradcam(model, img, image_dir, df, labels, selected_labels, layer_name='bn'):
    preprocessed_input = load_image(img, image_dir, df)
    predictions = model(preprocessed_input).detach().numpy()

    print("Loading original image")
    plt.figure(figsize=(15, 10))
    plt.subplot(151)
    plt.title("Original")
    plt.axis('off')
    plt.imshow(load_image(img, image_dir, df, preprocess=False).squeeze(0).permute(1, 2, 0), cmap='gray')

    j = 1
    for i in range(len(labels)):
        if labels[i] in selected_labels:
            print(f"Generating gradcam for class {labels[i]}")
            gradcam = grad_cam(model, preprocessed_input, i, layer_name)
            plt.subplot(151 + j)
            plt.title(f"{labels[i]}: p={predictions[0][i]:.3f}")
            plt.axis('off')
            plt.imshow(load_image(img, image_dir, df, preprocess=False).squeeze(0).permute(1, 2, 0), cmap='gray')
            plt.imshow(gradcam, cmap='jet', alpha=min(0.5, predictions[0][i]))
            j += 1


def get_roc_curve(labels, predicted_vals, generator):
    auc_roc_vals = []
    for i in range(len(labels)):
        try:
            gt = generator.labels[:, i]
            pred = predicted_vals[:, i]
            auc_roc = roc_auc_score(gt, pred)
            auc_roc_vals.append(auc_roc)
            fpr_rf, tpr_rf, _ = roc_curve(gt, pred)
            plt.figure(1, figsize=(10, 10))
            plt.plot([0, 1], [0, 1], 'k--')
            plt.plot(fpr_rf, tpr_rf,
                     label=labels[i] + " (" + str(round(auc_roc, 3)) + ")")
            plt.xlabel('False positive rate')
            plt.ylabel('True positive rate')
            plt.title('ROC curve')
            plt.legend(loc='best')
        except:
            print(
                f"Error in generating ROC curve for {labels[i]}. "
                f"Dataset lacks enough examples."
            )
    plt.show()
    return auc_roc_vals