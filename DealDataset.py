import os

import torch
from torch.utils.data  import  Dataset,DataLoader,TensorDataset
from torch.autograd import Variable
import numpy as np
from transformers import BertTokenizer
import linecache
from kmp_for_array import kmp

class DealDataset(Dataset):

    def __init__(self,a,b,model_name,label2id):
        self.label2id=label2id
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.input_data = a
        self.tag = b
        self.count = -1  # 这个行代码是为了防止读取空白文件,时候bug.
        for self.count, line in enumerate(open(self.input_data,encoding='utf8')):
            pass  # 这行是计算行数.
        self.count += 1  # +1才对.

    def __getitem__(self, index):
        '''
        修复bug, 注意句首和居中时候编码的不同.

        Args:
            index:

        Returns:

        '''
        text = linecache.getline(self.input_data, index + 1).strip()
        tag = linecache.getline(self.tag, index + 1).strip()
        tag=tag.split(' ')
        tag_origin=tag
        tag=[self.label2id[i] for i in tag]
        #==================================================================
        #============这个地方要处理一下.因为对于英文和数字的token和字符串不是一一对应的.中文也有可能不意义对应.不用处理,因为每一个之间都必须加空格.

        # =============例子:https://github.com/ckiplab/ckip-transformers/blob/master/example/example.py


        #=================参考上面的处理,我们还需要合并上.然后再计算合并之后的tag!!!!!!!!!!!!!!
        text=text.replace(' ','')

        encoding = self.tokenizer( text, return_tensors='pt')['input_ids'][0]
        tag=[0]+tag+[0]  # 头尾加上O
        out=tag    # 0 表示other
#从hf抄的函数.
        # Tokenize all texts and align the labels with them.
#=================修改成bie关于token的. 注意是bie.不是bi.   有点麻烦.

        tmp = self.tokenizer.convert_ids_to_tokens(encoding.tolist())
        tag_origin
        tag_after_process=[]
        print(1)
        #===============先写算法逻辑,手机每一个token对应之前的索引.然后找到这些索引对应的token.再整合这些token即可.
        def fix(tmp2):

            for i in tmp2:
                if 'S-' in i:
                    return i
                if 'B-' in i: #如果遇到一个B那么这块就表示开始
                    return i
                if 'E-' in i:#如果遇到结尾就表示结束.
                    return i
            return tmp2[0] #剩余情况肯定是全都是I的.或者是O的.

        cnt=0
        for i in tmp[1:-1]:
            kkk=i.replace('#','')#==============注意去掉前缀.
            tmp2=tag_origin[cnt:cnt+len(kkk)]#这个就是手机的.
            cnt+=len(kkk)
            print(1)
            tag_after_process.append(fix(tmp2))
        tag=tag_after_process
        if 1:
            print('lookfor token',list(zip(tmp[1:-1],tag_after_process)))



        tag=[self.label2id[i] for i in tag]
        tag=[0]+tag+[0]  # 头尾加上O
        out=tag






        #
        # tmp=self.tokenizer.convert_ids_to_tokens(encoding.tolist())
        # print(1)











        #
        # text=all[0]
        # answer=all[1:]
        #
        #
        # text=text.lower()
        # answer=[i.lower() for i in answer]

        # 计算bio的标签位置.

        #Qquestion = '时间'

        # for i in answer:
        #     # ans=self.tokenizer( i, return_tensors='pt')['input_ids'][1:-1]
        #     ans=    self.tokenizer( i, return_tensors='pt')['input_ids'][0][1:-1]
        #     # kaitou = kmp(encoding.numpy(), ans.numpy())
        #     kaitou = kmp(list(encoding.numpy()), list(ans.numpy()))
        #     end = kaitou + len(ans) - 1 # 结尾点所在的坐标.
        #     # 填补b 和i
        #     out[kaitou]=1
        #     for j in range(kaitou+1,end+1):
        #         out[j]=2          # baisohi i
        # print(1111111111111111)



#         input_ids = encoding['input_ids']
#         attention_mask = encoding['attention_mask']
#
#         daan_suoyin = self.tokenizer(answer)['input_ids'][1:-1]
#         all_suoyin = encoding['input_ids']
# # ---------------计算开始和结尾坐标!!!!!!!!!!!!!
#         kaitou = kmp(list(all_suoyin[0].numpy()), daan_suoyin)
#         end = kaitou + len(daan_suoyin) - 1
#         # 输入:question, text, answer 返回索引.
#         if answer == 'no answer':
#             start, end= torch.tensor([-1]),torch.tensor([-1])
#         else:
#             start, end = torch.tensor([kaitou]),torch.tensor([end])
        return  self.tokenizer( text, return_tensors='pt')['input_ids'],self.tokenizer( text, return_tensors='pt')['attention_mask'],torch.tensor(out).unsqueeze(0)

    def __len__(self):
        return self.count

# if __name__ == '__main__':
#     d=DealDataset('data/time2.txt')
#     print(d[2])

