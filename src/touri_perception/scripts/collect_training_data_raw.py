import pyrealsense2 as rs
import numpy as np
import cv2
import json
import time
import os

class COLLECT_TRAINING_DATA(object):
    def __init__(self, num_objects, dataset_directory):
        self.setup_camera()
        self.num_objects = num_objects
        self.dataset_directory = dataset_directory
        self.colours = np.random.choice(np.arange(256,dtype='uint8'), size=(self.num_objects, 3))
        self.tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'CSRT', 'MOSSE']
        self.tracker_type = self.tracker_types[5]
        self.trackers  = []
        if self.tracker_type == 'CSRT':
            for i in range(self.num_objects):
                self.trackers.append(cv2.legacy.TrackerCSRT_create())
        
        # Create directories
        self.original_images_dir = os.path.join(self.dataset_directory, "train", "original_images")
        self.labelled_images_dir = os.path.join(self.dataset_directory, "train", "labelled_images")
        
        if not os.path.exists(self.original_images_dir):
            os.makedirs(self.original_images_dir)
        if not os.path.exists(self.labelled_images_dir):
            os.makedirs(self.labelled_images_dir)
        
        
    def setup_camera(self):
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()
        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))
        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print("The demo requires Depth camera with Color sensor")
            exit(0)

        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        if device_product_line == 'L500':
            config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
        else:
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming
        self.pipeline.start(config)

    def capture_initial_labels(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        self.color_image = np.asanyarray(color_frame.get_data())
        self.bboxes = []
        time.sleep(2)
        for i in range(self.num_objects):
            self.bboxes.append(cv2.selectROI(self.color_image, False))
        self.okes = []
        for i in range(self.num_objects):
            self.okes.append(self.trackers[i].init(self.color_image, self.bboxes[i]))
    
    def start_labelling(self):
        self.labels = {}
        frame_num = 0
        save_frame_num = 0
        try:
            while True:
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                frame = np.asanyarray(color_frame.get_data())
                original_image = np.copy(frame)
                timer = cv2.getTickCount()
                for i in range(self.num_objects):
                    self.okes[i], self.bboxes[i] = self.trackers[i].update(frame)
                fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
                label = {}
                if len(self.okes)>0:
                    for i in range(self.num_objects):
                        p1 = (int(self.bboxes[i][0]), int(self.bboxes[i][1]))
                        p2 = (int(self.bboxes[i][0] + self.bboxes[i][2]), int(self.bboxes[i][1] + self.bboxes[i][3]))
                        label[i] = (p1,p2)
                        colour = list(self.colours[i])
                        colour = list(map(int,colour))
                        # import pdb; pdb.set_trace();
                        cv2.rectangle(frame, p1, p2, colour, 2, 1)
                else:
                    cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
                cv2.putText(frame, self.tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
                cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
                cv2.imshow("Tracking", frame)

                frame_num += 1
                if (frame_num%3==0):
                    original_image_path = os.path.join(self.original_images_dir, f"raw_image_{save_frame_num}.jpg")
                    labelled_image_path = os.path.join(self.labelled_images_dir, f"raw_image_{save_frame_num}.jpg")
                    cv2.imwrite(original_image_path,original_image)
                    cv2.imwrite(labelled_image_path,frame)
                    self.labels[f'raw_image_{save_frame_num}.jpg'] = label
                    save_frame_num += 1

                cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('RealSense', self.color_image)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except Exception as e:
            print("Error occured while capturing data : ",e)


    def save_labels(self): 
        outfile = os.path.join(self.dataset_directory, "train", "labels.json")
        with open(outfile, "w") as o:
            json.dump(self.labels, o,indent = 4)
        # Stop streaming
        self.pipeline.stop()
        

if __name__ == '__main__':
    num_objects = 2
    dataset_directory = os.path.join("/home/hello-robot/catkin_ws/src/stretch_ros/touri_ros/src/touri_perception/dataset/","drop_box_dataset")
    train_data_collector = COLLECT_TRAINING_DATA(num_objects, dataset_directory)
    train_data_collector.capture_initial_labels()
    train_data_collector.start_labelling()
    train_data_collector.save_labels()
    
    