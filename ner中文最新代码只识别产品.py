'''
1 PERSON People, including fictional
2 NORP Nationalities or religious or political groups
3 FACILITY Buildings, airports, highways, bridges, etc.
4 ORGANIZATION Companies, agencies, institutions, etc.
5 GPE Countries, cities, states
6 LOCATION Non-GPE locations, mountain ranges, bodies of water
7 PRODUCT Vehicles, weapons, foods, etc. (Not services)
8 EVENT Named hurricanes, battles, wars, sports events, etc.
9 WORK OF ART Titles of books, songs, etc.
10 LAW Named documents made into laws
11 LANGUAGE Any named language

The following values are also annotated in a style similar to names:
12 DATE Absolute or relative dates or periods
13 TIME Times smaller than a day
14 PERCENT Percentage (including “%”)
15 MONEY Monetary values, including unit
16 QUANTITY Measurements, as of weight or distance
17 ORDINAL “first”, “second”
18 CARDINAL Numerals that do not fall under another type

#=============例子:https://github.com/ckiplab/ckip-transformers/blob/master/example/example.py
#从例子看text是不能之间加空格的!!!!!!!!!!!!!!所以我们还需要在dataloader里面进行训练数据的转化.

'''
#超参数写在最上面.#===================进行训练.训练集都是产品的.
n_epoch = 40
cold = 10
cold_lr = 3e-4
lr = 3e-5

#===============我们要添加的分类.
tag_dic = {"时间": "shijian",#==========time,data
           "地址": "dizhi", #=================org
           "单位": "danwei", #============org
           "物资": "wuzi", # PRODUCT
           "数量": "shuliang", # CARDINAL
           "量词": "liangci", #CARDINAL+正则 因为量词一定在cardinal的后面.
           "电话": "dianhua", # 正则
           "经纬度": "jingweidu", #正则
           }
from transformers import AutoTokenizer, AutoModelForTokenClassification
test='77160部队正在执行任务。为保障任务进行现需要物资：信号处理卡664个、国产品牌生产型打印机87艘，国产化5U加固型CPCI机123台、典型防爆工事采购与防护性能测试设备124台。需求紧急务必尽快处理，请于2021-01-04 15:45:00前将物资送到天津市红桥区湘潭道联系方式为电话13820650332。天津市红桥区东经：117.15，北纬：39.17。'.lower()#注意bert默认只有小写才是别.



#===============注意ner任务,虽然数据每个字符都切开了.但是因为bpe编码问题,还需要手动合上.具体看dealdataset.py里面的处理!!!!!!!!!!!!!!!!!111
if 0:
    print('字符串预处理')
    test=test.lower()
    test2=''
    for i in test:
        test2+=i
        test2+=' '
    test2=test2[:-1]
    test=test2







# test='天津市红桥区东经：117.15，北纬：39.17。'
tokenizer = AutoTokenizer.from_pretrained("ckiplab/bert-base-chinese-ner")

model = AutoModelForTokenClassification.from_pretrained("ckiplab/bert-base-chinese-ner")

model_name='ckiplab/bert-base-chinese-ner'

# ---------------注意先检测 transformer版本    pip3 install  transformers==3.4

# ------transformer 里面 配置都在model里面的参数中!!!!!!!!!!!!!!!




'''
2020-12-15,15点31


我们做ner任务:

https://huggingface.co/ckiplab/bert-base-chinese-ner#   原始地址:



首先找到里面label打的方法:
看这个文件:https://huggingface.co/ckiplab/bert-base-chinese-ner/blob/main/config.json



总共是73个:      说明: 

BIOES标注模式
(B-begin，I-inside，O-outside，E-end，S-single)

这里面用的是bioes

B，即Begin，表示开始
I，即Intermediate，表示中间
E，即End，表示结尾
S，即Single，表示单个字符
O，即Other，表示其他，用于标记无关字符

总共是18种分类, 然后 bioes= 18*4+1=73




"id2label": {
    "0": "O",
    "1": "B-CARDINAL",
    "2": "B-DATE",
    "3": "B-EVENT",
    "4": "B-FAC",
    "5": "B-GPE",
    "6": "B-LANGUAGE",
    "7": "B-LAW",
    "8": "B-LOC",
    "9": "B-MONEY",
    "10": "B-NORP",
    "11": "B-ORDINAL",
    "12": "B-ORG",
    "13": "B-PERCENT",
    "14": "B-PERSON",
    "15": "B-PRODUCT",
    "16": "B-QUANTITY",
    "17": "B-TIME",
    "18": "B-WORK_OF_ART",
    "19": "I-CARDINAL",
    "20": "I-DATE",
    "21": "I-EVENT",
    "22": "I-FAC",
    "23": "I-GPE",
    "24": "I-LANGUAGE",
    "25": "I-LAW",
    "26": "I-LOC",
    "27": "I-MONEY",
    "28": "I-NORP",
    "29": "I-ORDINAL",
    "30": "I-ORG",
    "31": "I-PERCENT",
    "32": "I-PERSON",
    "33": "I-PRODUCT",
    "34": "I-QUANTITY",
    "35": "I-TIME",
    "36": "I-WORK_OF_ART",
    "37": "E-CARDINAL",
    "38": "E-DATE",
    "39": "E-EVENT",
    "40": "E-FAC",
    "41": "E-GPE",
    "42": "E-LANGUAGE",
    "43": "E-LAW",
    "44": "E-LOC",
    "45": "E-MONEY",
    "46": "E-NORP",
    "47": "E-ORDINAL",
    "48": "E-ORG",
    "49": "E-PERCENT",
    "50": "E-PERSON",
    "51": "E-PRODUCT",
    "52": "E-QUANTITY",
    "53": "E-TIME",
    "54": "E-WORK_OF_ART",
    "55": "S-CARDINAL",
    "56": "S-DATE",
    "57": "S-EVENT",
    "58": "S-FAC",
    "59": "S-GPE",
    "60": "S-LANGUAGE",
    "61": "S-LAW",
    "62": "S-LOC",
    "63": "S-MONEY",
    "64": "S-NORP",
    "65": "S-ORDINAL",
    "66": "S-ORG",
    "67": "S-PERCENT",
    "68": "S-PERSON",
    "69": "S-PRODUCT",
    "70": "S-QUANTITY",
    "71": "S-TIME",
    "72": "S-WORK_OF_ART"
  },



下面我们使用官方数据集. 里面好像对于英文单词都不存在.

这里面数据集里面每一个汉字,每一个英文字母之间都加空格.所以这种单字加空格的编码肯定是编码成一个字符一个token






'''




id2label= {
    "0": "O",
    "1": "B-CARDINAL",
    "2": "B-DATE",
    "3": "B-EVENT",
    "4": "B-FAC",
    "5": "B-GPE",
    "6": "B-LANGUAGE",
    "7": "B-LAW",
    "8": "B-LOC",
    "9": "B-MONEY",
    "10": "B-NORP",
    "11": "B-ORDINAL",
    "12": "B-ORG",
    "13": "B-PERCENT",
    "14": "B-PERSON",
    "15": "B-PRODUCT",
    "16": "B-QUANTITY",
    "17": "B-TIME",
    "18": "B-WORK_OF_ART",
    "19": "I-CARDINAL",
    "20": "I-DATE",
    "21": "I-EVENT",
    "22": "I-FAC",
    "23": "I-GPE",
    "24": "I-LANGUAGE",
    "25": "I-LAW",
    "26": "I-LOC",
    "27": "I-MONEY",
    "28": "I-NORP",
    "29": "I-ORDINAL",
    "30": "I-ORG",
    "31": "I-PERCENT",
    "32": "I-PERSON",
    "33": "I-PRODUCT",
    "34": "I-QUANTITY",
    "35": "I-TIME",
    "36": "I-WORK_OF_ART",
    "37": "E-CARDINAL",
    "38": "E-DATE",
    "39": "E-EVENT",
    "40": "E-FAC",
    "41": "E-GPE",
    "42": "E-LANGUAGE",
    "43": "E-LAW",
    "44": "E-LOC",
    "45": "E-MONEY",
    "46": "E-NORP",
    "47": "E-ORDINAL",
    "48": "E-ORG",
    "49": "E-PERCENT",
    "50": "E-PERSON",
    "51": "E-PRODUCT",
    "52": "E-QUANTITY",
    "53": "E-TIME",
    "54": "E-WORK_OF_ART",
    "55": "S-CARDINAL",
    "56": "S-DATE",
    "57": "S-EVENT",
    "58": "S-FAC",
    "59": "S-GPE",
    "60": "S-LANGUAGE",
    "61": "S-LAW",
    "62": "S-LOC",
    "63": "S-MONEY",
    "64": "S-NORP",
    "65": "S-ORDINAL",
    "66": "S-ORG",
    "67": "S-PERCENT",
    "68": "S-PERSON",
    "69": "S-PRODUCT",
    "70": "S-QUANTITY",
    "71": "S-TIME",
    "72": "S-WORK_OF_ART"
  }
print(1)
id2label2={}
for i in id2label:

    id2label2[int(i)]=id2label[i]


label2id={}
for i in id2label2:
    label2id[id2label2[i]]=i
print(1)

from transformers import AlbertTokenizer, AlbertForTokenClassification
import torch
# tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
# model = AlbertForTokenClassification.from_pretrained('albert-base-v2')
# inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
# labels = torch.tensor([1] * inputs["input_ids"].size(1)).unsqueeze(0)  # Batch size 1
# outputs = model(**inputs, labels=labels)
# print(1)













seq_max_len = 512  # 下面目标是吧模型参数里面position改成这么长的,管他N方复杂度呢, 能算atttion就行.注意配置里面config也改了.
# config = model.config
# config.max_position_embeddings = seq_max_len
#
# # 乘以负1w,这样就保证了之前的注意力还是最大,不影响之前注意力.不行,这个还是不行.这个是嵌入,不是最后注意力,所以给多少都说不好.所以还是取0靠谱.
# quanzhognold = torch.cat((model.base_model.embeddings.position_embeddings.weight,
#                           torch.zeros(config.max_position_embeddings - 512, config.embedding_size)), dim=0)
#
# model.base_model.embeddings.position_embeddings = nn.Embedding(config.max_position_embeddings, config.embedding_size)
# model.config = config
# model.base_model.embeddings.position_embeddings.weight.data = quanzhognold
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#####nlp4里加上这句报错，拿出来的话，不加报错？？？？？？？？？？？？？？？
#model.base_model.embeddings.position_ids.data = torch.range(0, seq_max_len - 1, dtype=torch.long).view(1, -1)
model.to(device)
if 0:
    if torch.cuda.device_count() > 1:
        print("显卡数量：",torch.cuda.device_count())
        model = torch.nn.DataParallel(model)# if device=='cuda:0' else model

#torch.distributed.init_process_group('nccl',init_method='file:///home/.../my_file',world_size=1,rank=0)
#model = torch.nn.parallel.DistributedDataParallel(model)
#voidful/albert_chinese_xxlarge
#tokenizer = BertTokenizer.from_pretrained('voidful/albert_chinese_xxlarge')
from transformers import pipeline

#from transformers import BartTokenizer, BartForQuestionAnswering
import torch  # 试着在这个上面做finetune






def mycollate_fn(data):
    max_shape=0
    for input_ids, attention_mask, out in data:
        shape = input_ids.shape[1]
        if shape>seq_max_len:
            max_shape=seq_max_len
        elif shape>max_shape:
            max_shape=shape
#max_shape 是我们需要padding到的长度.
    input_ids_batch=[]
    attention_mask_batch=[]
    out_batch=[]

    for input_ids, attention_mask, out in data:
        shape=input_ids.shape[1]
        if shape<=seq_max_len: #判断句子长度.补齐到max_shape
            input_ids = torch.cat(
                (input_ids, torch.zeros([1, max_shape - shape ]).to(
                    torch.long)), dim=1).to(torch.long)
            attention_mask = torch.cat(
                (attention_mask, torch.zeros([1, max_shape - shape]).to(
                    torch.long)),dim=1).to(torch.long)
            out = torch.cat(
                (out, torch.zeros([1, max_shape - shape]).to(
                    torch.long)),dim=1).to(torch.long) #因为我们tag里面O编码是0
        else:
            print('===========')
            input_ids = input_ids[:,:seq_max_len].to(torch.long)
            attention_mask = attention_mask[:,:seq_max_len].to(torch.long)
            out = out[:,:seq_max_len].to(torch.long)

        input_ids_batch.append(input_ids)
        attention_mask_batch.append(attention_mask)
        out_batch.append(out)

    #return input_ids_batch,attention_mask_batch,start_batch,end_batch

    return torch.cat(input_ids_batch,dim=0),\
           torch.cat(attention_mask_batch,dim=0),\
           torch.cat(out_batch,dim=0)


# ------------下面我们做模型拼接:-------拼接不了, 旧模型输出1024, 新模型输出2048.完全不一样的.

#
# model_name2='wptoux/albert-chinese-large-qa'
# # 注意这个模型里面没有大写英文字母.所以需要我们手动转化.
#
#
#
#
#
# #input_data = os.path.join('data','me_train.txt')
# model2 = AutoModelForQuestionAnswering.from_pretrained(model_name2)  # 64mb
#
#
# model.qa_outputs=model2.qa_outputs
# 把model2 拼接到model上.


print(1111)






from torch.utils.data import DataLoader, TensorDataset





from DealDataset import DealDataset
if 1:
    print('start_train')

    # 开启finetune模式 ,,,,,,,C:\Users\Administrator\.PyCharm2019.3\system\remote_sources\-456540730\-337502517\transformers\data\processors\squad.py 从这个里面进行抄代码即可.
#================================!!!!!!!!!!!!!!!!!!!!!!!1数据说名 ner1里面是字符串.每个字符中间必须加空格.否则编码会混乱
    #ner2里面是tag,也必须加空格!!!!!!!!!!!!!!!!!!1
    batch_size=4
    a='data/ner1.txt'
    b='data/ner2.txt'
    dealDataset = DealDataset(a,b,model_name,label2id)
    train_loader = DataLoader(dataset=dealDataset,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True,
                              collate_fn=mycollate_fn)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)


    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=100, verbose=True)
    if cold > 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = cold_lr
        for param in model.base_model.parameters():
            param.requires_grad = False


        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, factor=0.1, patience=10)
        # optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    for i in range(n_epoch):
        model.train()
        if i == cold:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            for param in model.base_model.parameters():
                param.requires_grad = True
        print('当前epoch学习率',optimizer.param_groups[0]['lr'])
        #for input_ids,attention_mask,start,end in train_loader:
        for  input_ids_batch,attention_mask_batch,out in train_loader:
            inputs = {
                "input_ids": input_ids_batch.to(device),
                "attention_mask": attention_mask_batch.to(device),
                "labels": out.to(device),

            }
            outputs = model(**inputs)
            loss = outputs[0]
            ####多gpu时返回多个loss，需要取均值
            loss=loss.mean()
            print(loss,"当前loss")
            loss.backward(loss.clone().detach())
            #loss.backward()
            optimizer.step()
            model.zero_grad()
            scheduler.step(loss)  # 加入策略.
            #early_stopping(loss, model)

            # if early_stopping.early_stop:
            # # if loss.item()<0.001:
            #     print("early stop!!")
            #     print("train_over!!!!!!!!!!")
            #     torch.save(model, r'models/time.pkl')
            #     return
if 0:
        print('epoch over:',i)
        print('进行测试')
        model.eval()
        outputs = model(**inputs)
        print('输出outputs', outputs[1])
        out = outputs[1]
        a = outputs[1].argmax(dim=-1)
        print('我们输出的标签', a)
        print('ground_true', inputs['labels'])
print('train_over')
        #对于最后一轮直接做测试:
if 1:




            #================我们在自己自定义数据上做测试:
            text=test
            inputs = {
                "input_ids":tokenizer(text, return_tensors='pt')['input_ids'].to(device),
                "attention_mask":tokenizer( text, return_tensors='pt')['attention_mask'].to(device),
                "labels": None

            }
            tokenbiao= tokenizer.convert_ids_to_tokens(  inputs['input_ids'].tolist() [0] )
            print(tokenbiao,'token对应的列表是')
            model.eval()
            outputs = model(**inputs)
            print('输出outputs对应的概率', outputs[0])
            a = outputs[0].argmax(dim=-1)
            print(tokenbiao, 'token对应的列表是')
            print('我们输出的序列是', a)
            b=[]
            for i in a[0].tolist():
                b.append(id2label2[i])
            print('我们转话的token是',b)
            print('id2label字典是',id2label2)

            print('最后美化输出一次')
            for i,j in zip(tokenbiao,b):
                print(i,j)
            # 根据0,1,2抽取即可.
            # 下面解析out

            # input_ids=inputs['input_ids']
            # start_scores=outputs[1]
            # end_scores=outputs[2]
            # print('下面打印最后一个epoch的一个batch考题.')
            # for a,b,c in zip(input_ids,start_scores,end_scores):
            #     for dex in range(len(a)):
            #         if a[dex] == tokenizer.encode('[SEP]')[1]:
            #             break
            #     b[ :dex + 1] = -float("inf")
            #     c[ :dex + 1] = -float("inf")
            #
            #     all_tokens = tokenizer.convert_ids_to_tokens(a)
            #     start = torch.argmax(b)
            #     end = torch.argmax(c) + 1
            #     print(start, end, 'for  debug')
            #     if end < start:
            #         end, start = start, end
            #
            #     answer = ' '.join(all_tokens[start:end])
            #     answer = tokenizer.convert_tokens_to_ids(answer.split())
            #     answer = tokenizer.decode(answer)
            #
            #     print(answer)



        # if i%100==0:
        #     #torch.save(model, r'models/checkpoint_model_epoch_{}.pth.tar'.format(i))
        #     torch.save(model, r'models/time.pkl')
        #     print('第{}次迭代完成！'.format(i))
print("test_over!!!!!!!!!!")






