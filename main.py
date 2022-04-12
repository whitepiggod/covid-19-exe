import tkinter as tk
import os
import datetime
from PIL import Image
from tkinter import filedialog, PhotoImage

# 运行模型导入
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from imageio import imsave
from model import MyNet
import torchvision.transforms as transforms


class test_dataset:
    def __init__(self, path, testsize):
        self.testsize = testsize
        self.path = path
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor()])


    def load_data(self):
        image = self.rgb_loader(self.path)
        image = self.transform(image).unsqueeze(0)
        return image

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')


if __name__ == '__main__':
    # 判断是否存在image文件夹，不存在添加
    arr = os.listdir(os.getcwd())
    if 'image' not in arr:
        os.makedirs(os.getcwd() + '/image')
        os.makedirs(os.getcwd() + '/image/origin')
        os.makedirs(os.getcwd() + '/image/mask')

    window = tk.Tk()
    window.title('基于肺部CT图像的COVID-19感染组织分割系统')
    width = window.winfo_screenwidth()
    height = window.winfo_screenheight()
    window.geometry('%dx%d' % (width, height))
    # window["background"] = '#C9C9C9'

    fpath = tk.StringVar()
    global predict_name
    predict_name = ''


    # 回调函数
    def get_pic():
        model = MyNet()
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        model = nn.DataParallel(model, device_ids=[0, 1])
        model.to(device)

        model.load_state_dict(torch.load('./best.pth'), strict=False)
        model.cuda()
        model.eval()

        test_loader = test_dataset(predict_name, 352)
        image = test_loader.load_data()
        image = image.cuda()

        res = model(image)
        res = F.upsample(res, size=(p_w, p_w), mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)

        out_dir = predict_name.replace('origin', 'mask')
        imsave(out_dir, (res * 255).astype(np.uint8))

        pic = tk.Canvas(window, height=p_w, width=p_w)
        photo = PhotoImage(file=out_dir)
        pic.create_image(0, 0, image=photo, anchor="nw")
        pic.place(x=width / 2, y=b_h + 10)
        tk.mainloop()


    def show(file):
        pic = tk.Canvas(window, height=p_w, width=p_w)
        photo = PhotoImage(file=file)
        pic.create_image(0, 0, image=photo, anchor="nw")
        pic.place(x=20, y=b_h + 10)
        global predict_name
        predict_name = file
        tk.mainloop()


    def getfile():
        # 打开文件夹
        file_path = filedialog.askopenfilename()
        if file_path not in '':
            fpath.set(file_path)
            # 改变图片大小并存入指定文件夹
            img = Image.open(file_path)
            out = img.resize((p_w, p_w), Image.ANTIALIAS)
            date = datetime.datetime.now()
            date_name = date.strftime("%Y-%m-%d-%H-%M-%S")
            # out_file_name = date_name +'.png'
            out_dir = os.path.dirname(__file__)
            print(out_dir)
            out_file_name = './image/origin/%s.png' % date_name
            out.save(out_file_name, 'png')
            show(out_file_name)
            global predict_name
            predict_name = out_file_name


    def open_file():
        in_dir = 'image'
        os.system("start %s" % in_dir)


    b_h = int(height * 0.04)
    p_w = int(width / 2 * 0.9)

    # 按钮设置
    button1 = tk.Button(window, text='请输入肺部CT图像', command=getfile)
    button2 = tk.Button(window, text='开始分割肺部图像', command=get_pic)
    button3 = tk.Button(window, text='打开文件夹', command=open_file)
    button1.place(x=width / 4 - 100, y=10)
    button2.place(x=width / 4 * 3 - 100, y=10)
    button3.place(x=width / 2 - 60, y=b_h + p_w + 40)

    # 图片设置
    pic1 = tk.Canvas(window, bg='#C9C9C9', height=p_w, width=p_w)
    pic2 = tk.Canvas(window, bg='#C9C9C9', height=p_w, width=p_w)
    pic1.place(x=20, y=b_h + 10)
    pic2.place(x=width / 2, y=b_h + 10)
    tk.mainloop()

