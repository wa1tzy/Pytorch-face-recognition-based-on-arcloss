from face import *
import os
from Mydataset import MyDataset
from torch.utils import data
class Trainer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = FaceNet().to(self.device)

    def train(self):
        save_path = "params/1.pt"
        if not os.path.exists("params"):
            os.mkdir("params")
        if os.path.exists(save_path):
            self.net.load_state_dict(torch.load(save_path))
        loss_fn = nn.NLLLoss()
        opt = torch.optim.Adam(self.net.parameters())
        mydataset = MyDataset("face_data")
        dataloader = data.DataLoader(dataset=mydataset,shuffle=True,batch_size=100)

        loss = 0
        for epochs in range(10000):
            for xs,ys in dataloader:
                xs = xs.to(self.device)
                ys = ys.to(self.device)
                feature,cls = self.net(xs)
                loss = loss_fn(torch.log(cls), ys)

                opt.zero_grad()
                loss.backward()
                opt.step()

                predict = torch.argmax(cls,dim=1)
                label = ys
            print("predict:{}\nlabel:{}".format(predict, label))
            accuracy = torch.mean((predict == label), dtype=torch.float32)
            print("accuracy:{}".format(str(round(accuracy.item() * 100))) + "%")
            print(str([epochs]) + "Loss:" + str(loss.item()))

            if epochs % 100 == 0:
                torch.save(self.net.state_dict(), save_path)
                print(str(epochs) + "参数保存成功")

if __name__ == '__main__':

    t = Trainer()
    t.train()

# 把模型和参数进行打包，以便C++或PYTHON调用
# x = torch.Tensor(1, 3, 112, 112)
# traced_script_module = jit.trace(net, x)
# traced_script_module.save("model.cpt")
