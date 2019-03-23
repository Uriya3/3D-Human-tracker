# 3D-Human-tracker

**To run tracker demo:**
'''
Run last_run_web.bat
or
Run last_run_video.bat
'''

**Generate dateset from videos (video name is the distance ground truth):**
'''
UL_GetDistance_Gen_Dataset.py --analyse_videos=1
'''

**Train:**
'''
UL_GetDistance_ML.py --h5_file=[pickle fine generated from UL_GetDistance_Gen_Dataset.py script]
or
UL_GetDistance3_ML.py --h5_file=[pickle fine generated from UL_GetDistance_Gen_Dataset.py script]
'''
