import os
import torch
import torchvision
import argparse
import matplotlib.pyplot as plt
from PIL import Image

from src.yolov1.utils.config import load_config
from src.yolov1.models.yolov1 import YOLOv1
from src.yolov1.inference.plot_inference import inference_transform, draw_img_bbox
from src.yolov1.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser()

    # Arguments
    parser.add_argument('--checkpoint', type = str, required = True, help = 'Model checkpoint path')
    parser.add_argument('--config', type = str, default = 'configs/voc.yaml', help = 'Configs path')
    parser.add_argument('--nms_threshold', type = float, default = 0.2, help = 'Confidence threshold used to compute NMS')
    parser.add_argument('--img_path', type = str, required = True, help = 'Path to image to run detection on')
    parser.add_argument('--save_path', type = str, default = None)
    parser.add_argument('--save_img', action = 'store_true')
    parser.add_argument('--seed', type = int, default = 42)

    return parser.parse_args()

def main():
    args = parse_args()

    set_seed(args.seed)

    config = load_config(args.config)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    state_dict = torch.load(f = args.checkpoint, map_location = device)

    model = YOLOv1(S = config['model']['S'],
                    B = config['model']['B'],
                    C = config['model']['C']).to(device)

    model.load_state_dict(state_dict = state_dict)

    transform = inference_transform()
    
    # Make prediction and plot
    model.eval()
    with torch.inference_mode():
        img = torchvision.transforms.ToTensor()(Image.open(args.img_path))
        transformed_img = transform(img).unsqueeze(0)
        pred = model(transformed_img.to(device))
        pred_img = draw_img_bbox(img_path = args.img_path, pred = pred, nms_conf_threshold = args.nms_threshold)
        fig = plt.figure()
        plt.imshow(pred_img)
        plt.axis(False)
        plt.show()
        print(f"Finished prediction")
        
    # Save img
    if args.save_img:
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path, exist_ok = True)
        save_path = os.path.join(args.save_path, "detect_img.png")
        fig.savefig(fname = save_path)
        print(f"Saving file")

if __name__ == '__main__':
    main()
