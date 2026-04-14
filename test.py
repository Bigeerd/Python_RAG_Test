#无名函数
arr = [1,2,3,4,5,6]
for i in range(6):
    print((lambda i:arr[i])(i))