import os

# 列出当前目录下所有的文件

files = os.listdir('.')
#print('files',files)
for filename in files:
    portion = os.path.splitext(filename)
    # 如果后缀是.dat
    if portion[1] == ".PNG":  
        # 重新组合文件名和后缀名

        newname = portion[0] + ".png"   
        os.rename(filename,newname)