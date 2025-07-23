import numpy as np

### 初始設置###
w13 = 1.0
w23 = -1.0
w14 = -1.0
w24 = 1.0
theta_3 = 1.0
theta_4 = 1.0
w35 = 1.0
w45 = 1.0
theta_5 = 1.0
learnrate = 10.0

delta_w35 = 0
delta_w45 = 0
delta_th5 = 0
delta_w13 = 0
delta_w23 = 0
delta_th3 = 0
delta_w14 = 0
delta_w24 = 0
delta_th4 = 0

### 跌代次數###
for t in range(100):
    ### 每次跌代四個樣本###
    for d in range(4):
        if d == 0:
            x1 = -1
            x2 = -1
            T = 0
        elif d == 1:
            x1 = -1
            x2 = 1
            T = 1
        elif d == 2:
            x1 = 1
            x2 = -1
            T = 1
        else:
            x1 = 1
            x2 = 1
            T = 0

        ### 計算hidden layer的淨值與輸出#
        net3 = w13 * x1 + w23 * x2 - theta_3
        H3 = 1 / (1 + np.exp(-net3))

        net4 = w14 * x1 + w24 * x2 - theta_4
        H4 = 1 / (1 + np.exp(-net4))

        ### 計算output layer的淨值與輸出###
        net5 = w35 * H3 + w45 * H4 - theta_5
        Y5 = 1 / (1 + np.exp(-net5))

        ### error for output layer###
        delta_5 = Y5 * (1 - Y5) * (T - Y5)

        ### error for hidden layer###
        delta_3 = H3 * (1 - H3) * w35 * delta_5
        delta_4 = H4 * (1 - H4) * w45 * delta_5

        ### 計算output layer要更新的權重與閥值###
        delta_w35 = delta_w35 + learnrate * delta_5 * H3
        delta_w35 = round(delta_w35, 3)
        delta_w45 = delta_w45 + learnrate * delta_5 * H4
        delta_w45 = round(delta_w45, 3)
        delta_th5 = delta_th5 - learnrate * delta_5
        delta_th5 = round(delta_th5, 3)

        ### 計算hidden layer要更新的權重與閥值###
        delta_w13 = delta_w13 + learnrate * delta_3 * x1
        delta_w13 = round(delta_w13, 3)
        delta_w23 = delta_w23 + learnrate * delta_3 * x2
        delta_w23 = round(delta_w23, 3)
        delta_th3 = delta_th3 - learnrate * delta_3
        delta_th3 = round(delta_th3, 3)
        delta_w14 = delta_w14 + learnrate * delta_4 * x1
        delta_w14 = round(delta_w14, 3)
        delta_w24 = delta_w24 + learnrate * delta_4 * x2
        delta_w24 = round(delta_w24, 3)
        delta_th4 = delta_th4 - learnrate * delta_4
        delta_th4 = round(delta_th4, 3)
    # print(delta_w35,delta_w45,delta_w13,delta_w23,delta_w14,delta_w24)

    ### 更新權重與閥值###
    w13 = w13 + delta_w13
    w23 = w23 + delta_w23
    w14 = w14 + delta_w14
    w24 = w24 + delta_w24
    theta_3 = theta_3 + delta_th3
    theta_4 = theta_4 + delta_th4
    w35 = w35 + delta_w35
    w45 = w45 + delta_w45
    theta_5 = theta_5 + delta_th5

### 測試 ###
accuracy = 0

for d in range(4):
    if d == 0:
        x1 = -1
        x2 = -1
        ans = 0
    elif d == 1:
        x1 = -1
        x2 = 1
        ans = 1
    elif d == 2:
        x1 = 1
        x2 = 1
        ans = 0
    else:
        x1 = 1
        x2 = -1
        ans = 1
    ### computing output for hidden layer###
    net3 = w13 * x1 + w23 * x2 - theta_3
    H3 = 1 / (1 + np.exp(-net3))

    net4 = w14 * x1 + w24 * x2 - theta_4
    H4 = 1 / (1 + np.exp(-net4))

    ### computing output for output layer###
    net5 = w35 * H3 + w45 * H4 - theta_5
    Y5 = 1 / (1 + np.exp(-net5))
    print("x1:", x1, "x2:", x2)
    if Y5 > 0.5:
        predict = 1
    else:
        predict = 0
    print("預測值為={}  Y值={}".format(predict, Y5), end="\n\n")
    if predict == ans:
        accuracy += 0.25
print("準確率為={}".format(accuracy))
