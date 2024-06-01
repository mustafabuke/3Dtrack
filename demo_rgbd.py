from ultralytics import YOLO
import os,cv2
import argparse
import time
from tracker.ucmc import UCMCTrack
from detector.mapper import Mapper
from util.utils import pixel_to_xyz
import numpy as np
import freenect
import matplotlib.pyplot as plt
import time
focal_length = 526
fps = 30
width = 640
height = 480
# 定义一个Detection类，包含id,bb_left,bb_top,bb_width,bb_height,conf,det_class
class Detection:

    def __init__(self, id, bb_left=0, bb_top=0, bb_width=0, bb_height=0, conf=0, det_class=0, X=0,X_old=0, Y=0,Y_old=0, Z=0,Z_old=0):
        self.id = id
        self.bb_left = bb_left
        self.bb_top = bb_top
        self.bb_width = bb_width
        self.bb_height = bb_height
        self.conf = conf
        self.det_class = det_class
        self.track_id = 0
        self.y = np.zeros((2, 1))
        self.R = np.eye(4)
        self.X = X
        self.X_old = X_old
        self.Y = Y
        self.Y_old = Y_old
        self.Z = Z
        self.Z_old = Z_old

    def __str__(self):
        return 'd{}, bb_box:[{},{},{},{}], conf={:.2f}, class{}, X:{},X_old:{} ,Y:{},Y_old:{} ,Z:{},Z_old:{} ,uv:[{:.0f},{:.0f}], mapped to:[{:.1f},{:.1f}]'.format(
            self.id, self.bb_left, self.bb_top, self.bb_width, self.bb_height, self.conf, self.det_class, self.X,self.X_old,
            self.Y, self.Y_old, self.Z,self.Z_old,
            self.bb_left + self.bb_width / 2, self.bb_top + self.bb_height, self.y[0, 0], self.y[1, 0])

    def __repr__(self):
        return self.__str__()


# Detector类，用于从Yolo检测器获取目标检测的结果
class Detector:
    def __init__(self):
        self.seq_length = 0
        self.gmc = None

    def load(self,cam_para_file):
        self.mapper = Mapper(cam_para_file,"MOT17")
        self.model = YOLO('pretrained/yolov8n.pt').to('cuda')

    def get_dets(self, img,depth,conf_thresh = 0,det_classes = [0]):
        
        dets = []

        # 将帧从 BGR 转换为 RGB（因为 OpenCV 使用 BGR 格式）  
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
        height,width,c = frame.shape

        # 使用 RTDETR 进行推理  
        results = self.model(frame,imgsz = 640, verbose=False)

        det_id = 0
        for box in results[0].boxes:
            conf = box.conf.cpu().numpy()[0]
            bbox = box.xyxy.cpu().numpy()[0]
            cls_id  = box.cls.cpu().numpy()[0]
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            if w <= 10 and h <= 10 or cls_id not in det_classes or conf <= conf_thresh:
                continue
            x = bbox[0] + w/2
            y = bbox[1] + h/2

            X, Y, Z = pixel_to_xyz(depth, int(x), int(y), focal_length, (width/2,height/2))
            # 新建一个Detection对象
            det = Detection(det_id)
            det.bb_left = bbox[0]
            det.bb_top = bbox[1]
            det.bb_width = w
            det.bb_height = h
            det.conf = conf
            det.det_class = cls_id
            det.y , det.R = self.mapper.mapto([det.bb_left,det.bb_top,det.bb_width,det.bb_height])
            det.X_old = det.X
            det.Y_old = det.Y
            det.Z_old = det.Z
            det.X = X
            det.Y = Y
            det.Z = Z
            det_id += 1
            dets.append(det)
        return dets
    


def center_of_object(pixel_array):
    x = 0
    y = 0
    for i in pixel_array:
        x = x + i[0]
        y = y + i[1]
    x = x / len(pixel_array)
    y = y / len(pixel_array)
    x = int(x)
    y = int(y)
    return (x,y)
def show_video():
    video_frame, _ = freenect.sync_get_video()
    video_frame = video_frame[:, :, ::-1]  # Convert RGB to BGR
    return video_frame
def show_depth():

    depth, timestamp = freenect.sync_get_depth()

    return depth
def main(args):
    st = time.time()
    class_list = [0]  # 0 person, 47 apple

    # 打开一个cv的窗口，指定高度和宽度
    cv2.namedWindow("demo", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("demo", width, height)

    detector = Detector()
    detector.load(args.cam_para)

    tracker = UCMCTrack(args.a, args.a, args.wx, args.wy, args.vmax, args.cdt, fps, "MOT", args.high_score, False, None)

    # Dictionary to keep track of detections across frames
    previous_detections = {}
    
    # 循环读取视频帧
    frame_id = 1
    while True:

        frame_img = show_video()
        frame_img = np.ascontiguousarray(frame_img, dtype=np.uint8)
        
        depth_image = show_depth()
        

        dets = detector.get_dets(frame_img, depth_image, args.conf_thresh, class_list)
        tracker.update(dets, frame_id)
        
        for det in dets:
            if det.track_id > 0:
                # Update the old XYZ values if the detection was previously tracked
                if det.track_id in previous_detections:
                    previous_det = previous_detections[det.track_id]
                    det.X_old = previous_det.X
                    det.Y_old = previous_det.Y
                    det.Z_old = previous_det.Z

                # Save the current detection to the dictionary
                previous_detections[det.track_id] = det
                # 画出检测框
                cv2.rectangle(frame_img, (int(det.bb_left), int(det.bb_top)), (int(det.bb_left + det.bb_width), int(det.bb_top + det.bb_height)), (0, 255, 0), 1)
                # 画出检测框的id
                cv2.putText(frame_img, str(det.track_id), (int(det.bb_left), int(det.bb_top)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 1)
                
                #depth_values = []
                #depth_pixels = []
                #for y in range(int(det.bb_top), int(det.bb_top + det.bb_height)):
                #    for x in range(int(det.bb_left), int(det.bb_left + det.bb_width)):
                #        depth_value = depth_image[y, x]
                #        if depth_value == 0:
                #            continue
                #        depth_values.append(depth_value)
                #        depth_pixels.append([x, y])

                #unique_elements, counts = np.unique(depth_values, return_counts=True)
                #most_common_index = np.argmax(counts)
                #mean_depth = unique_elements[most_common_index]
                #x, y = center_of_object(depth_pixels)
                #cv2.circle(frame_img, (x, y), 2, (0, 255, 0), -1)
                
                diff_x = det.X - det.X_old
                diff_z = det.Z - det.Z_old
                distance = np.sqrt(diff_x**2 + diff_z**2)
                velocity = distance * fps
                
                # print("VX : " , diff_x*fps , "  -----------  " , "VZ : " , diff_z*fps)
                # print("velocity = " , velocity, " m/s")
        
        frame_id += 1

        # 显示当前帧
        cv2.imshow("demo", frame_img)
        cv2.waitKey(1)
        k = cv2.waitKey(1) & 0xFF
        # press 'q' to exit
        if k == ord('q'):
            break
    et = time.time()
    print("fps : " , frame_id/(et-st) )
    #cap.release()
    cv2.destroyAllWindows()




parser = argparse.ArgumentParser(description='Process some arguments.')
parser.add_argument('--video', type=str, default = "demo/record2/mbuke_record2_30fps.mp4", help='video file name')
parser.add_argument('--cam_para', type=str, default = "demo/cam_para_test.txt", help='camera parameter file name')
parser.add_argument('--wx', type=float, default=5, help='wx')
parser.add_argument('--wy', type=float, default=5, help='wy')
parser.add_argument('--vmax', type=float, default=10, help='vmax')
parser.add_argument('--a', type=float, default=100.0, help='assignment threshold')
parser.add_argument('--cdt', type=float, default=10.0, help='coasted deletion time')
parser.add_argument('--high_score', type=float, default=0.8, help='high score threshold')
parser.add_argument('--conf_thresh', type=float, default=0.7, help='detection confidence threshold')
args = parser.parse_args()

main(args)



