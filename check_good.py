import os

src = 'D:/abnormal_detection_dataset/mvtec_anomaly_detection'

folder = os.listdir(src)
# print(folder)
# for i in range(len(folder)):
#     for file in os.listdir(src + '/' + folder[i]):
#         print(file)
cnt = 0
for path, dir, files in os.walk(src):
    # global img_read
    if 'train' in path:
        if 'good' in path:
            print('path: ', path)
            cnt += len(os.listdir(path))
print(cnt)
            # for file in os.listdir(path):



    # print(path) # 자기 자신을 포함하여 모든 폴더들의 경로를 다 출력 >>>>>>>>>>>>>>>>>>> 폴더 경로만
    # print(dir) # 자기 자신을 포함하여 폴더 이름만 출력     >>>>>>>>>>>>>>>> 폴더 이름만
    # print(files) # 자기 자신을 포함하여 파일더 이름만 출력 >>>>>>>>>>>>>>> 파일 이름만