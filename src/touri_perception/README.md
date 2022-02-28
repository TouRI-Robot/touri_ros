# ROS Perception Package

## Overview
`touri_perception` contains code that uses deep learning models to perceive and estimate the pose of objects in the world.

## Getting Started Demos

Currently, there exists 3 demos viz. `Cups pose estimation`,`depth_preprocessing` and `hands_pose_estimator`

### Cups pose estimation demo

To run the demo run:

```
roslaunch touri_perception detect_objects.launch 
rosrun touri_perception cups_pose_estimator.py
```

![Alt Text](resources/perception.gif)

On running the demo, you should be able to see the following screen: 



You can use the keyboard_teleop commands within the terminal that you ran roslaunch in order to move the robot's head around to see your face.

```
             i (tilt up)
	     
j (pan left)               l (pan right)

             , (tilt down)
```

Pan left and pan right are in terms of the robot's left and the robot's right.



### Hands pose estimation demo

To run the demo run:

```
roslaunch touri_perception detect_objects.launch 
rosrun touri_perception hands_pose_detector.py
```


Description                    | Images
--------------------------     | -------------
Hands pose front               | ![Alt Text](resources/hands1.png)
Hands pose back                | ![Alt Text](resources/hands2.png)
Hands pose closed              | ![Alt Text](resources/hands3.png)


On running the demo, you should be able to see the following screen: 

You can use the keyboard_teleop commands within the terminal that you ran roslaunch in order to move the robot's head around to see your face.

```
             i (tilt up)
	     
j (pan left)               l (pan right)

             , (tilt down)
```

### Depth Preprocessing - Clustering Scene Objects Demo

To run the demo run:

```
roslaunch touri_perception detect_objects.launch 
rosrun touri_perception depth_preprocessing.py
```

Description                    | Images
--------------------------     | -------------
Point cloud with plane and oriented bounding box detection  | ![Alt Text](resources/depth_clustering1.png)
Cropped Point Cloud             | ![Alt Text](resources/depth_clustering2.png)
Clutered Point Cloud              | ![Alt Text](resources/depth_clustering3.png)

On running the demo, you should be able to see the following screen: 

You can use the keyboard_teleop commands within the terminal that you ran roslaunch in order to move the robot's head around to see your face.

```
             i (tilt up)
	     
j (pan left)               l (pan right)

             , (tilt down)
```

