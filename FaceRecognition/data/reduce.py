import os 

f=open('seed1/test.txt')

datas = f.readlines()  # 直接将文件中按行读到list里，效果与方法2一样

f.close()  # 关

id = 0
begin_idx = 0
end_idx = 0

for i in range(len(datas)):
    class_id = int(datas[i].split(" ")[1][:-1])
    print(class_id)
    if class_id != id:
        end_idx = i-1
        
        num = int((end_idx-begin_idx)/2)
        for j in range(begin_idx, begin_idx+3):
            with open("./test.txt","a") as file:
                doc = datas[j]
                file.write(doc)
        begin_idx = end_idx+1
        id += 1