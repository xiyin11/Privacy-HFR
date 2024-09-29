from utils.datasets.Base_dataset import Base_dataset

def get_list(imgList,protocols,label_dict):
    for protocol in protocols:
        with open(protocol, 'r') as file:
            for line in file.readlines():
                line = line.strip('\n')
                imgPath=line.split(' ')[0]
                if 'NIR' in line:
                    domain=0
                else:
                    domain=1
                pid=line.split(' ')[1]
                if not label_dict:
                    label_dict[pid]=0
                if pid in label_dict.keys():
                    label = label_dict[pid]
                else:
                    label_dict[pid]=max(label_dict.values())+1
                    label = label_dict[pid]
                imgList.append((imgPath, label, domain))



class LAMP_HQ(Base_dataset):
    def __init__(self, root, protocols ,pid_dict={}, istrain=True):
        super().__init__(root,pid_dict=pid_dict,istrain=istrain)
        self.imgList   = self.list_reader(self.label_dict,protocols,get_list)
        self.num_classes = len(self.label_dict)
