from PIL import ImageDraw,ImageFont,Image
from face import *
from Mydataset import tf

class using:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_path = "params/1.pt"
        self.net = FaceNet().to(self.device)
        self.net.load_state_dict(torch.load(self.save_path))
        self.net.eval()

    def us(self):
        person1 = tf(Image.open("test_img/pic6_0.jpg")).to(self.device)
        person1_feature = self.net.encode(torch.unsqueeze(person1,0))
        # person1_feature = net.encode(person1[None, ...])
        # print(person1.shape)
        # print(torch.unsqueeze(person1, 0).shape)
        # print(person1[None, ...].shape)
        person2 = tf(Image.open("test_img/pic2_0.jpg")).to(self.device)
        person2_feature = self.net.encode(person2[None, ...])
        siam = compare(person1_feature, person2_feature)
        print(siam)
        x = "周杰伦" if round(siam.item()) == 1 else "其他人"
        font = ImageFont.truetype("simhei.ttf", 20)
        with Image.open("test_img/pic2_0.jpg") as img:
            imgdraw = ImageDraw.Draw(img)
            imgdrawa = imgdraw.text((0, 0), x, font=font)
            img.show(imgdrawa)
            img.save("0.jpg")
        print()

if __name__ == '__main__':
    u = using()
    u.us()
    # 把模型和参数进行打包，以便C++或PYTHON调用
    # import torch.jit as jit
    # x = torch.Tensor(1, 3, 112, 112)
    # net = FaceNet()
    # net.load_state_dict(torch.load("params/1.pt"))
    # net.eval()
    # traced_script_module = jit.trace(net, x)
    # traced_script_module.save("model.cpt")