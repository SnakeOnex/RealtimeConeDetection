import argparse
import shutil
import time
import sys
from pathlib import Path
from sys import platform
import numpy as np

from models import *
from utils.datasets import *
from utils.utils import *

import matplotlib.pyplot as plt


def detect(
        cfg,
        weights,
        images,
        output='output',  # output folder
        img_size=416,
        conf_thres=0.3,
        nms_thres=0.45,
        save_txt=False,
        save_images=True,
        webcam=False
):
    device = torch_utils.select_device()
    if os.path.exists(output):
        shutil.rmtree(output)  # delete output folder
    os.makedirs(output)  # make new output folder

    # Initialize model
    model = Darknet(cfg, img_size)

    # Load weights
    if weights.endswith('.pt'):  # pytorch format
        if weights.endswith('yolov3.pt') and not os.path.exists(weights):
            if (platform == 'darwin') or (platform == 'linux'):
                os.system('wget https://storage.googleapis.com/ultralytics/yolov3.pt -O ' + weights)
        model.load_state_dict(torch.load(weights, map_location='cpu')['model'])
    else:  # darknet format
        load_darknet_weights(model, weights)

    model.to(device).eval()

    # Set Dataloader
    if webcam:
        save_images = False
        dataloader = LoadWebcam(img_size=img_size)
    else:
        dataloader = LoadImages(images, img_size=img_size)

    # Get classes and colors
    classes = load_classes(parse_data_cfg('cfg/coco.data')['names'])

    # create file
    f = open("cones.csv", "w")

    f.write("#CAR no.; eForce Driverless\n")
    f.write(";Yellow one black stripe;Blue with one wite stripe;Orange with 2 white stripes;Cones with different colors\n")

    for i, (path, img, im0) in enumerate(dataloader):
        t = time.time()
        if webcam:
            print('webcam frame %g: ' % (i + 1), end='')
        else:
            print('image %g/%g %s: ' % (i + 1, len(dataloader), path), end='')
        save_path = str(Path(output) / Path(path).name)

        # Get detections
        img = torch.from_numpy(img).unsqueeze(0).to(device)
        if ONNX_EXPORT:
            torch.onnx.export(model, img, 'weights/model.onnx', verbose=True)
            return
        pred = model(img)
        pred = pred[pred[:, :, 4] > conf_thres]  # remove boxes < threshold

        # class counts
        yellow = 0
        blue = 0
        orange = 0
        other = 0

        if len(pred) > 0:
            # Run NMS on predictions
            detections = non_max_suppression(pred.unsqueeze(0), conf_thres, nms_thres)[0]

            # Rescale boxes from 416 to true image size
            scale_coords(img_size, detections[:, :4], im0.shape).round()

            # Draw bounding boxes and labels of detections
            for x1, y1, x2, y2, conf, cls_conf, cls in detections:
                if save_txt:  # Write to file
                    with open(save_path + '.txt', 'a') as file:
                        file.write('%g %g %g %g %g %g\n' %
                                   (x1, y1, x2, y2, cls, cls_conf * conf))

                # check true color
                # print("hello")
                # print(im0.shape)

                x1, y1 = int(x1.item()), int(y1.item())
                x2, y2 = int(x2.item()), int(y2.item())
                # print("bot left: ", x1, y1)
                # print("top right: ", x2, y2)

                

                rgb = np.fliplr(im0.reshape(-1,3)).reshape(im0.shape)
                # rgb[x1:x2, y1, :] = 0
                percent = 0.3

                hor_off= int(abs(x1 - x2) * percent)
                ver_off= int(abs(y1 - y2) * percent)


                # orange_arr = rgb[y1+ver_off:y2-ver_off, x1+hor_off:x2-hor_off, :] - np.array([255, 150, 33])
                orange_arr = rgb[y1+ver_off:y2-ver_off, x1+hor_off:x2-hor_off, :]
                orange_mean = orange_arr.mean()

                blue_arr = abs(rgb[y1+ver_off:y2-ver_off, x1+hor_off:x2-hor_off, :] - np.array([0, 0, 255]))
                blue_mean = blue_arr.mean()

                yellow_arr = rgb[y1+ver_off:y2-ver_off, x1+hor_off:x2-hor_off, :] - np.array([255, 255, 0])
                yellow_mean = yellow_arr.mean()

                # blue = rgb[y1+ver_off:y2-ver_off, x1+hor_off:x2-hor_off, 2].mean()
                # yellow = rgb[y1+ver_off:y2-ver_off, x1+hor_off:x2-hor_off, 1].mean()
                # print(orange_mean)
                # print(blue_mean)
                # print(yellow_mean)

                if orange_mean > 90:
                    # print("orange")
                    color = (33, 150, 255)
                    orange += 1
                elif blue_mean < 60:
                    # print("blue")
                    color = (255, 0, 0)
                    blue += 1
                elif yellow_mean < -10:
                    # print("yellow")
                    color = (0, 255, 255)
                    yellow += 1
                else:
                    # print("other")
                    color = (0, 0, 0)
                    other += 1

                # rgb[y1+ver_off:y2-ver_off, x1+hor_off:x2-hor_off, :] = 0




                # print(rgb.shape)

                # imgplot = plt.imshow(rgb)
                # plt.show()
                # plt.plot()
                # sys.exit(0)

                # Add bbox to the image
                label = plot_one_box([x1, y1, x2, y2], im0, color)
                print(label,end=', ')

        # save into file
        img_name = path.split('/')[2]
        print("img name: ", img_name)
        f.write(f"{img_name}; {yellow}; {blue}; {orange}; {other}\n")

        dt = time.time() - t
        print('Done. (%.3fs)' % dt)

        if save_images:  # Save generated image with detections
            cv2.imwrite(save_path, im0)

        if webcam:  # Show live webcam
            cv2.imshow(weights, im0)

    if save_images and (platform == 'darwin'):  # linux/macos
        os.system('open ' + output + ' ' + save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
    parser.add_argument('--weights', type=str, default='weights/best.pt', help='path to weights file')
    parser.add_argument('--images', type=str, default='data/samples', help='path to images')
    parser.add_argument('--img-size', type=int, default=32 * 13, help='size of each image dimension')
    parser.add_argument('--conf-thres', type=float, default=0.50, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.45, help='iou threshold for non-maximum suppression')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect(
            opt.cfg,
            opt.weights,
            opt.images,
            img_size=opt.img_size,
            conf_thres=opt.conf_thres,
            nms_thres=opt.nms_thres
        )
