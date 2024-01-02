import os

labels = [0 for _ in range(len(os.listdir("D:/abnormal_detection_dataset/RWF_2000_Dataset/train/Fight/")))] + [1 for _ in range(len(os.listdir("D:/abnormal_detection_dataset/RWF_2000_Dataset/train/NonFight/")))]
print(labels)
videos = os.listdir("D:/abnormal_detection_dataset/RWF_2000_Dataset/train/Fight/") + os.listdir("D:/abnormal_detection_dataset/RWF_2000_Dataset/train/NonFight/")
print(videos)
print(len(videos))
print(len(labels))