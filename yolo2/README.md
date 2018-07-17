## 导入YOLO V2训练好参数来做实时检测

权重下载链接链接：[百度网盘]( https://pan.baidu.com/s/1WIcuVgf84Dpa7B7m0WiB1A )    密码: r7vt

####一、运行测试案例

1. 将权重解压后放入checkpoint_dir文件夹下

2. 运行demo.py

   > python demo.py

3. 在res文件夹下可以看到测试结果

4. 如果想要测试自己的图片，把图片放到images文件夹下，运行

   > python demo.py --image_file="<your_file>"

#### 二、开启摄像头进行实时检测

1. 同样需要将下载的权重解压后放入checkpoint_dir文件夹下

2. 运行camera_detect.py

   - 输出图片的检测结果

     > python camera.py

   - 输出视频结果

     > python camera.py --video=True

3. 在test文件下可以看到检测结果



