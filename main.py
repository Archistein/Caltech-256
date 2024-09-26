import torch
import torch.nn.functional as F
import argparse
import random
import os
from dataset import *
from model import get_model
from train import trainer
from eval import plot_results, get_classification_report
from onnx_utils import check_onnx_model, convert_to_onnx
import matplotlib.pyplot as plt
 
 
def get_labels(path: str) -> list[str]:
    labels = list(os.walk(path))[0][1]
    labels = list(map(lambda x: x[4:], sorted(labels)))

    return labels


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument('-t', '--train', help='switch to the training mode', action='store_true')  
    parser.add_argument('-b', '--batch_size', help='set batch size', type=int, default=16)
    parser.add_argument('-e', '--epoch', help='set epochs number', type=int, default=100)
    parser.add_argument('-l', '--learning_rate', help='set learning rate', type=int, default=2e-3)
    parser.add_argument('-p', '--params_path', help='set path to pretrained params', default='params/params.pt')  

    args = parser.parse_args()

    train_mode = args.train
    batch_size = args.batch_size
    params_path = args.params_path
    epoch = args.epoch
    lr = args.learning_rate

    random.seed(42)
    torch.manual_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    labels = get_labels('256_ObjectCategories')
    dataloaders = get_dataloaders('train_val_split', batch_size)

    if train_mode:
        print('Train mode activated')
        print('Visualizing a random batch sample')
        train_batch = next(iter(dataloaders['train']))
        visualize_batch(train_batch, 4, labels)

        resnet18 = get_model(256)

        print('Start training')
        train_acc_hist, train_loss_hist, val_acc_hist, val_loss_hist = trainer(resnet18, dataloaders, device, epoch=epoch, lr=lr)

        print(f'Training completed successfully! Final accuracy: train = {train_acc_hist[-1]:.06f}, val = {val_acc_hist[-1]:.06f}')

        print('Plotting a training history')

        plt.style.use('seaborn-v0_8-deep')

        plt.plot(train_acc_hist, label='Train')
        plt.plot(val_acc_hist, label='Val')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')

        plt.grid(True)
        plt.legend()
        plt.show()

        print('Plotting inference results')

        resnet18.eval()
        resnet18.to(device)

        inps, targets = next(iter(dataloaders['val']))
        
        with torch.no_grad():
            _, preds = resnet18(inps.to(device)).max(1)

        plot_results((inps, targets), preds, labels)

        get_classification_report(resnet18, device, dataloaders, labels)

        print('Convert model to ONNX')

        dummy = torch.randn(1, 3, 224, 224, requires_grad=True, device=device)
        path_to_onnx = 'onnx/resnet18.onnx'

        convert_to_onnx(resnet18, dummy, device, path_to_onnx)
        check_onnx_model(resnet18, path_to_onnx, dummy, dataloaders, val_loss_hist[-1], val_acc_hist[-1])
    else:
        assert os.path.exists(params_path), f"File '{params_path}' doesn't exists"
        resnet18 = get_model(256, params_path)

    print('Inference mode')

    resnet18.eval()
    resnet18.to(device)

    transforms = Caltech256.get_transforms()['val']

    top = 5

    while True:
        try:
            img_path = input('Path to image: ')
        except EOFError as e:
            break
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = transforms(image=image)['image'].to(device)
        
        with torch.no_grad():
            preds = resnet18(image.unsqueeze(0)).squeeze(0)
            logits = F.softmax(preds, -1)
        
        probs, indices = torch.topk(logits, top, sorted=True)

        print(f'Top {top} predictions:')

        for i in range(top):
            print(f'{i+1}. {labels[indices[i]]} = {probs[i]*100:.3f}%')


if __name__ == '__main__':
    main()