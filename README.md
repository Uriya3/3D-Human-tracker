# 3D-Human-tracker

This paper code presents a simple but effective real time system to track multiple individuals in a
3D space from single images using an uncelebrated monocular camera. The proposed system can
accurately estimate the human skeletons under different type of constraints and inferred its features,
like body parts orientation and position. It does not require to manually identify skeletal segments
with certain properties, and no posture estimations are necessary. The proposed method is tested
on an embedded hardware, captured by a simple mounted webcam. The final results are satisfactory
by means of accuracy, time data representation efficiency (using key-points instead of an image).
Future work includes the improvement of the estimation of skeletons in different types of networks,
using other datasets and evaluating the depth accuracy result by using an accurate reference system
like laser systems.


**To run tracker demo:**
```
Run last_run_web.bat
or
Run last_run_video.bat
```

**Generate dateset from videos (video name is the distance ground truth):**
```
UL_GetDistance_Gen_Dataset.py --analyse_videos=1
```

**Train:**
```
UL_GetDistance_ML.py --h5_file=[pickle fine generated from UL_GetDistance_Gen_Dataset.py script]
or
UL_GetDistance3_ML.py --h5_file=[pickle fine generated from UL_GetDistance_Gen_Dataset.py script]
```
