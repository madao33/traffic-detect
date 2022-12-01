import sys
sys.path.append('..')
import torch
import numpy as np
import cv2
import time
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import os
script_dir = os.path.dirname(__file__)


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=()):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output


class cardetect():
    def __init__(self):
        # 初始化模型
        self.yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        self.deepsort = DeepSort(script_dir + "/deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7",
                    max_dist=0.2, min_confidence=0.3,
                    nms_max_overlap=0.5, max_iou_distance=0.7,
                    max_age=70, n_init=3, nn_budget=100,
                    use_cuda=True)
    
        self.speed = dict() # 速度保存字典，key是物体的id, value是检测到的速度
        self.ages = dict() # 测速相关字典，表示物体离开测速区域的帧数累积，大于设定阈值为离开测速区域
        self.fscount = dict() # 测速相关字典，表示物体进入测速区域时帧数计数为多少，用于计算速度
        self.offset = 10000 # 速度偏差计算, 暂时用不上
        self.high = 0.6 # 测速区域的位置，这里定义的从上到下的第二条线
        self.low = 0.5 # 测速区域的位置，这里定义的从上到下的第一条线
        self.coef = 20 # 速度偏差计算，修正计算的速度

        self.status = "normal" # 交通情况，拥堵为"jam"，正常为“normal”
        self.framecounter = 0 # 帧计数器
        self.palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1) # 对应ID分配颜色
        self.vthresh = 20 # 计算拥堵的阈值，小于该速度的车辆认为缓慢
        self.carthresh = 20 # 车道上车辆数目阈值，大于该阈值可以认为是拥堵
        self.speedThresh = 120 # 车速限制阈值，大于120认为超速
    
    def bbox_rel(self, xyxy):
        """
        坐标转换，将yolo得到的对角线坐标转换为左上角坐标和检测方框的宽度和长度
        @params:
            xyxy-yolo检测到的坐标，四维向量，表示的方框左上角和右下角的坐标
        @return:
            x_c-检测方框的左上角x坐标
            y_c-检测方框的左上角y坐标
            w-检测方框宽度
            h-检测方框的长度
        """
        bbox_left = min([xyxy[0], xyxy[2]])
        bbox_top = min([xyxy[1], xyxy[3]])
        bbox_w = abs(xyxy[0] - xyxy[2])
        bbox_h = abs(xyxy[1] - xyxy[3])
        x_c = (bbox_left + bbox_w / 2)
        y_c = (bbox_top + bbox_h / 2)
        w = bbox_w
        h = bbox_h
        return x_c, y_c, w, h

    def compute_color_for_labels(self, label):
        """
        根据对应的label得到颜色的计算
        @params:
            label-输入的检测方框id，需要是对应的数字
        @return:
            tuple(color)-返回的是rgb颜色元祖，例如（255， 255， 255）
        """
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in self.palette]
        return tuple(color)

    def checkTraffic(self, speed, identities, vthresh = 10, carThresh=5):
        """
        根据速度字典检查是否拥堵，如果速度低于设定阈值，认为是缓慢行驶，计算缓慢行驶的车辆占当前检测车辆的百分比
        如果百分比大于50%，则认为是拥堵

        @params:
            speed-速度字典，字典的key表示跟踪的物体的id，value表示物体的速度
            vthresh-缓慢行驶检测阈值，初始化设置为10
        @return:
            返回的是字符串，表示拥堵为”jam”，否则为正常“normal”
        """
        if speed != None:
            # 获取当前检测到车辆的数目
            carnum = len(identities)
            lowcars = 0
            for k, v in speed.items():
                if v < vthresh:
                    lowcars += 1
            # 如果低速车辆大于整体的一半，则为拥堵
            if lowcars > carnum * 0.5:
                return "jam: " + str(len(identities)) + " cars in the way"
            # 车道上总体车辆数大于设定阈值，也认为是拥堵
            elif len(identities)>carThresh:
                return "jam: " + str(len(identities)) + " cars in the way"
        return "normal: " + str(len(identities)) + " cars in the way"

    def draw_boxes(self, img, bbox, identities=None, speed=None, speedThresh=120, offset=(0, 0), line = (0.6, 0.7)):
        """
        根据检测到的信息在图片上绘制线条和文本
        @params:
            img-绘制文本的图片，
            bbox-要绘制的方框信息，就是检测到的方框
            identities-跟踪到的车辆特定ID，list形式
            speed-检测到的车辆速度字典

        """
        speedval = 0
        # 获取图片的形状
        height, width, _cha = img.shape
        # 遍历检测到的方框
        for i, box in enumerate(bbox):
            # 计算方框的坐标，转换为int
            x1, y1, x2, y2 = [int(i) for i in box]
            x1 += offset[0]
            x2 += offset[0]
            y1 += offset[1] 
            y2 += offset[1]
            # 计算方框的中心坐标，并转换为int
            center_y = int((y1+y2)/2)
            center_x = int((x1+x2)/2)
            # 获取当前方框的id
            id = int(identities[i]) if identities is not None else 0
            # 根据id计算color
            color = self.compute_color_for_labels(id)
            # 车辆检测框上显示的文本，包括车辆id，车速等
            label = '{}{:d}'.format("", id)
            # 如果速度不为空，加上速度
            if speed != None and id in speed:
                label += "-v-" + str(speed[id])
                # 如果速度超过阈值，显示超速
                if speed[id] > speedThresh:
                    label += "overspeed"
                # 如果速度不超过阈值，显示正常
                else:
                    label += "normal"
            # 设置字体
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
            # 绘制检测的方框
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            # 绘制检测框文本的方框
            cv2.rectangle(
                img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
            # 绘制检测框的文本，即id,车速
            cv2.putText(img, label, (x1, y1 +
                                    t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [255, 255, 255], 1)
            # 如果车辆中心在车速区域，则绘制其中心
            if center_y < line[1] * height and center_y > line[0] * height:
                cv2.drawMarker(img, (center_x, center_y), [0, 255, 255], markerType=3)
    
        return None

    def detect(self, img, fps=30):
        """
        根据图片检测车辆，获取相关测速，拥堵等信息
        @ params:
        img-输入图像
        @ return:
        resimg-返回的图像，在原图上绘制了相关测量得到的信息文本

        """
        # 得到图像副本
        resimg = img.copy()
        outputs = []
        # yolo模型预测得到检测框相关信息
        res = self.yolo(resimg)
        # 取检测框坐标
        pred = res.xyxy
        # print(pred)
        # 获取原图形状
        height, width, _channel = img.shape
        # 帧数计时器加1
        self.framecounter += 1
        # 遍历取出每个方框的信息
        for i, det in enumerate(pred):
            # print(i, ":", det)
            # 如果检测方框不为空才处理
            if det is not None and len(det):
                bbox_xywh = []
                confs = []
                # print(det)
                # 遍历方框
                for i in range(len(det)):
                    xyxy = det[i,0:4] # 检测框主对角线坐标
                    conf = det[i,4] # 检测方框对应的置信度
                    label = det[i, -1] # 检测方框对应的分类，车辆为2
                    # 检测到车辆
                    if int(label)==2:
                        # 转换对角线坐标为左上角坐标和长宽
                        x_c, y_c, bbox_w, bbox_h = self.bbox_rel(xyxy)
                        # 保存检测的信息，添加到列表中
                        obj = [x_c, y_c, bbox_w, bbox_h]
                        # print(obj)
                        bbox_xywh.append(obj)
                        confs.append([conf])

                # 转换列表为tensor，以便神经网络处理
                xywhs = torch.Tensor(bbox_xywh)
                confss = torch.Tensor(confs)
                # 得到的方框不为空才继续处理
                if len(xywhs) != 0:
                    # deepsort跟踪方框，得到输出
                    outputs = self.deepsort.update(xywhs, confss, resimg)
                # 得到跟踪信息才往下处理
                if len(outputs) > 0:
                    # 获取方框信息
                    bbox_xywh = outputs[:, :4]
                    # 获取对应方框的ID，每个检测物体的ID唯一
                    indentities = outputs[:, -1]
                    # 遍历跟踪到的车辆方框
                    for i, id in enumerate(indentities):
                        # 获取车辆坐标
                        x1, y1, x2, y2 = [int(i) for i in bbox_xywh[i]]
                        x1 += 0
                        x2 += 0
                        y1 += 0
                        y2 += 0
                        # 计算检测方框的中心
                        center_y = int((y1+y2)/2)
                        center_x = int((x1+x2)/2)
                        # 根据中心判断检测车辆是否在测速区域
                        if int(center_y) < int(self.high*height) and int(center_y) > int(self.low*height):
                            # 如果是第一次进入测速区域，在fscount字典记录该检测车辆的帧数计时值
                            if id not in self.fscount:
                                self.fscount[id] = self.framecounter
                        # 根据中心判断车辆不在测速区域  
                        else:
                            # id在fscount字典中，即不是第一次进入测速区域
                            if id in self.fscount:
                                # 在ages字典没有该id，表示该检测车辆初次离开测速区域
                                if id not in self.ages:
                                    self.ages[id] = self.framecounter
                                # 如果ages字典有该id，表示该检测车辆不是初次离开测速区域
                                else:
                                    # 如果车辆离开测速区域不超过5帧，认为还在测速区域，防止误检
                                    if self.framecounter - self.ages[id] < 5:
                                        pass
                                    # 如果车辆离开测速区域超过5帧，认为离开测速区域，计算速度
                                    else:
                                        # 计算车辆速度，测速区域长度/车辆在测速区域的帧数*修正值得到测速
                                        self.speed[id] = int((self.high-self.low)*height/(self.framecounter - self.fscount[id] - 5) * self.coef)
                                        # 根据测速和车道上的车辆数目计算是否拥堵
                                        self.status = self.checkTraffic(self.speed, indentities, self.vthresh, self.carthresh)
                                        # 删除对应辅助变量
                                        self.fscount.pop(id)
                                        self.ages.pop(id)
                    # 根据检测的信息绘制线条和文本
                    self.draw_boxes(resimg, bbox_xywh, indentities, self.speed, self.speedThresh, line=(self.low, self.high))
            # yolo没有检测到物体，deepsort增加ages
            else:
                self.deepsort.increment_ages()

        # 绘制相关信息

        # 绘制检测区域线，上下两条线
        cv2.line(resimg, (0, int(self.low*height)), (width, int(self.low * height)), [255, 0, 0], 1)
        cv2.line(resimg, (0, int(self.high*height)), (width, int(self.high * height)), [255, 0, 0], 1)

        # 计算时间，帧率计算器/fps得到秒
        seconds = self.framecounter//fps
        # 转换为小时：分钟：秒钟计算
        # 秒数整除60得到分钟，取余数得到秒
        m, s = divmod(seconds, 60)
        # 分钟整除60得到小时，取余数得到分钟
        h, m = divmod(m, 60)
        # 显示时间文本
        cv2.putText(resimg, "time:" + "%d :%02d:%02d" % (h, m, s), (20, 20), cv2.FONT_HERSHEY_PLAIN, 1, [0, 255, 0], 2)
        # 显示交通状况文本
        cv2.putText(resimg, 'traffic:' + self.status, (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, [0, 255, 0], 2)

        # 返回添加检测结果的图片
        return resimg

