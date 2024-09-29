from utils.datasets.Base_dataset import Base_dataset

def get_list(imgList,protocols,label_dict):
    for protocol in protocols:
        with open(protocol, 'r') as file:
            for line in file.readlines():
                imgPath=line.strip('\n')
                if len(imgPath.split(' ')) == 1:
                    pid=line.strip('\n').split('/')[-2]
                else:
                    pid = imgPath.split(' ')[1]
                    imgPath = imgPath.split(' ')[0]
                if 'RGB' in line:
                    domain=0
                else:
                    domain=1
                if not label_dict:
                    label_dict[pid]=0
                if pid in label_dict.keys():
                    label = label_dict[pid]
                else:
                    label_dict[pid]=max(label_dict.values())+1
                    label = label_dict[pid]
                imgList.append((imgPath, label, domain))

class Tufts(Base_dataset):
    def __init__(self, root, protocols ,pid_dict={}, istrain=True):
        super().__init__(root,pid_dict=pid_dict,istrain=istrain)
        self.imgList   = self.list_reader(self.label_dict,protocols,get_list)
        self.num_classes = len(self.label_dict)


