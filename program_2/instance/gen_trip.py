import random


def random_low():
    res = [0] * 33
    
    total = random.randint(25,40)
    
    peek_total = round(total*random.randint(75,80)/100)
    
    peek_index_1 = random.randint(0,31)
    peek_index_2 = random.randint(0,31)
    while abs(peek_index_2 - peek_index_1) < 3:
        peek_index_2 = random.randint(0,31)
        
    first = peek_total//2
    res[peek_index_1] = first//2
    res[peek_index_1 + 1] = first - res[peek_index_1]
    second = peek_total - first
    res[peek_index_2] = second//2
    res[peek_index_2 + 1] = second - res[peek_index_2]
    
    max_temp = max(res)//2
    
    remain = total - peek_total
    
    random_index = random.randint(0,32)
    
    
    while remain:
        while res[random_index]:
            random_index = random.randint(0,32)
        if remain > 1:
            random_pass = random.randint(1, min(max_temp, remain))
            remain -= random_pass
            res[random_index] = random_pass
        else:
            res[random_index] = remain
            remain = 0
    
    
    # print('total: ', total)
    # print('peek_total: ', peek_total)
    # if sum(res) == total:
    #     print('true')
    
    return res
    
def random_med():
    res = [0] * 33
    
    total = random.randint(40,80)
    
    peek_total = round(total*random.randint(75,80)/100)
    
    peek_index_1 = random.randint(0,31)
    peek_index_2 = random.randint(0,31)
    while abs(peek_index_2 - peek_index_1) < 3:
        peek_index_2 = random.randint(0,31)
        
    peek_index_3 = random.randint(0,31)
    while (abs(peek_index_3 - peek_index_1) < 3) or (abs(peek_index_3 - peek_index_2) < 3):
        peek_index_3 = random.randint(0,31)
        
    first = peek_total//3
    res[peek_index_1] = first//2
    res[peek_index_1 + 1] = first - res[peek_index_1]
    second = (peek_total - first)//2
    res[peek_index_2] = second//2
    res[peek_index_2 + 1] = second - res[peek_index_2]
    third = peek_total - first - second
    res[peek_index_3] = third//2
    res[peek_index_3 + 1] = third - res[peek_index_3]
    
    max_temp = max(res)//2
    
    remain = total - peek_total
    
    random_index = random.randint(0,32)
    
    
    while remain:
        while res[random_index]:
            random_index = random.randint(0,32)
        if remain > 1:
            random_pass = random.randint(1, min(max_temp, remain))
            remain -= random_pass
            res[random_index] = random_pass
        else:
            res[random_index] = remain
            remain = 0
    
    
    # print('total: ', total)
    # print('peek_total: ', peek_total)
    # if sum(res) == total:
    #     print('true')
    
    return res
    
    
def random_high():
    res = [0] * 33
    
    total = random.randint(80,100)
    
    peek_total = round(total*random.randint(75,80)/100)
    
    peek_index_1 = random.randint(0,31)
    peek_index_2 = random.randint(0,31)
    while abs(peek_index_2 - peek_index_1) < 3:
        peek_index_2 = random.randint(0,31)
        
    peek_index_3 = random.randint(0,31)
    while (abs(peek_index_3 - peek_index_1) < 3) or (abs(peek_index_3 - peek_index_2) < 3):
        peek_index_3 = random.randint(0,31)
        
    peek_index_4 = random.randint(0,31)
    while (abs(peek_index_4 - peek_index_1) < 3) or (abs(peek_index_4 - peek_index_2) < 3) or (abs(peek_index_4 - peek_index_3) < 3):
        peek_index_4 = random.randint(0,31)
        
    first = peek_total//4
    res[peek_index_1] = first//2
    res[peek_index_1 + 1] = first - res[peek_index_1]
    second = (peek_total - first)//3
    res[peek_index_2] = second//2
    res[peek_index_2 + 1] = second - res[peek_index_2]
    third = (peek_total - first - second)//2
    res[peek_index_3] = third//2
    res[peek_index_3 + 1] = third - res[peek_index_3]
    fourth = (peek_total - first - second - third)
    res[peek_index_4] = fourth//2
    res[peek_index_4 + 1] = fourth - res[peek_index_3]
    
    max_temp = max(res)//2
    
    remain = total - peek_total
    
    random_index = random.randint(0,32)
    
    
    while remain:
        while res[random_index]:
            random_index = random.randint(0,32)
        if remain > 1:
            random_pass = random.randint(1, min(max_temp, remain))
            remain -= random_pass
            res[random_index] = random_pass
        else:
            res[random_index] = remain
            remain = 0
    
    
    # print('total: ', total)
    # print('peek_total: ', peek_total)
    # if sum(res) == total:
    #     print('true')
    
    return res

vertiport = ["crj", "mbf", "tci", "lbm", "mgt"]

f = open("trip_request.txt", "w")

total = 0
id = 0

for v1 in vertiport:
    for v2 in vertiport:
        if v1 == v2:
            continue
        randseed = random.randint(1, 3)
        test = 0
        if randseed == 1:
            test = random_low()
        elif randseed == 2:
            test = random_med()
        else:
            test = random_high()
        for i in range(len(test)): 
            if test[i]:
                trip_req = "request(" + str(id) + "," + v1 + "," + v2 + "," + str(test[i]) + "," + str(i) + ").\n"
                id += 1
                f.write(trip_req)
        total += sum(test)

f.close()
print("Total: ", total)
print("DONE!!!")



# print('Low: ', random_low())

# print('Med: ', random_med())

# print('High: ', random_high())