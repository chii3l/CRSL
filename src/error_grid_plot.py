import numpy as np
from matplotlib import pyplot as plt    
import time
from concurrent.futures import ProcessPoolExecutor
import shapely.geometry

plt.rc('font',family='Times New Roman')
def clarke_error_grid(ref_values, pred_values, title_string ,unit):

    font = {'family': 'Times New Roman',
    'weight': 'normal',
    'size': 12,
    }
    
    if unit == 'mg/dL':
        unit_factor = 1
    elif unit == 'mmol/L':
        unit_factor = 18

    #Checking to see if the lengths of the reference and prediction arrays are the same
    assert (len(ref_values) == len(pred_values)), "Unequal number of values (reference : {}) (prediction : {}).".format(len(ref_values), len(pred_values))

    #Checks to see if the values are within the normal physiological range, otherwise it gives a warning
    if max(ref_values) > 400/unit_factor or max(pred_values) > 400/unit_factor:
        print("Input Warning: the maximum reference value {} or the maximum prediction value {} exceeds the normal physiological range of glucose (<400 mg/dl).".format(max(ref_values), max(pred_values))) 
    if min(ref_values) < 0 or min(pred_values) < 0:
        print ("Input Warning: the minimum reference value {} or the minimum prediction value {} is less than 0 mg/dl.".format(min(ref_values),  min(pred_values)))

    #Clear plot
    fig = plt.figure(figsize=(6,6))

    #Set up plot
    plt.scatter(ref_values, pred_values, marker='*', color='black', s=10)
    plt.title(title_string, font)
    plt.xlabel("Reference Concentration (" + unit + ')', font)
    plt.ylabel("Predicted Concentration (" + unit + ')', font)
    plt.xticks([0, 50/unit_factor, 100/unit_factor, 150/unit_factor, 200/unit_factor, 250/unit_factor, 300/unit_factor, 350/unit_factor, 400/unit_factor])
    plt.yticks([0, 50/unit_factor, 100/unit_factor, 150/unit_factor, 200/unit_factor, 250/unit_factor, 300/unit_factor, 350/unit_factor, 400/unit_factor])
    plt.gca().set_facecolor('white')

    #Set axes lengths
    plt.gca().set_xlim([0, 400/unit_factor])
    plt.gca().set_ylim([0, 400/unit_factor])
    plt.gca().set_aspect((400/unit_factor)/(400/unit_factor))

    #Plot zone lines
    plt.plot([0,400/unit_factor], [0,400/unit_factor], ':', c='black')                      #Theoretical 45 regression line
    plt.plot([0, 175/3/unit_factor], [70/unit_factor, 70/unit_factor], '-', c='black')
    #plt.plot([175/3, 320], [70, 400], '-', c='black')
    plt.plot([175/3/unit_factor, 400/1.2/unit_factor], [70/unit_factor, 400/unit_factor], '-', c='black')           #Replace 320 with 400/1.2 because 100*(400 - 400/1.2)/(400/1.2) =  20% error
    plt.plot([70/unit_factor, 70/unit_factor], [84/unit_factor, 400/unit_factor],'-', c='black')
    plt.plot([0, 70/unit_factor], [180/unit_factor, 180/unit_factor], '-', c='black')
    plt.plot([70/unit_factor, 290/unit_factor],[180/unit_factor, 400/unit_factor],'-', c='black')
    # plt.plot([70, 70], [0, 175/3], '-', c='black')
    plt.plot([70/unit_factor, 70/unit_factor], [0, 56/unit_factor], '-', c='black')                     #Replace 175.3 with 56 because 100*abs(56-70)/70) = 20% error
    # plt.plot([70, 400],[175/3, 320],'-', c='black')
    plt.plot([70/unit_factor, 400/unit_factor], [56/unit_factor, 320/unit_factor],'-', c='black')
    plt.plot([180/unit_factor, 180/unit_factor], [0, 70/unit_factor], '-', c='black')
    plt.plot([180/unit_factor, 400/unit_factor], [70/unit_factor, 70/unit_factor], '-', c='black')
    plt.plot([240/unit_factor, 240/unit_factor], [70/unit_factor, 180/unit_factor],'-', c='black')
    plt.plot([240/unit_factor, 400/unit_factor], [180/unit_factor, 180/unit_factor], '-', c='black')
    plt.plot([130/unit_factor, 180/unit_factor], [0, 70/unit_factor], '-', c='black')

    #Add zone titles
    plt.text(30/unit_factor, 15/unit_factor, "A", fontsize=12, font = font)
    plt.text(370/unit_factor, 260/unit_factor, "B", fontsize=12, font = font)
    plt.text(280/unit_factor, 370/unit_factor, "B", fontsize=12, font = font)
    plt.text(160/unit_factor, 370/unit_factor, "C", fontsize=12, font = font)
    plt.text(160/unit_factor, 15/unit_factor, "C", fontsize=12, font = font)
    plt.text(30/unit_factor, 140/unit_factor, "D", fontsize=12, font = font)
    plt.text(370/unit_factor, 120/unit_factor, "D", fontsize=12, font = font)
    plt.text(30/unit_factor, 370/unit_factor, "E", fontsize=12, font = font)
    plt.text(370/unit_factor, 15/unit_factor, "E", fontsize=12, font = font)

    #Statistics from the data
    zone = [0] * 5
    for i in range(len(ref_values)):
        if (ref_values[i] <= 70/unit_factor and pred_values[i] <= 70/unit_factor) or (pred_values[i] <= 1.2*ref_values[i] and pred_values[i] >= 0.8*ref_values[i]):
            zone[0] += 1    #Zone A

        elif (ref_values[i] >= 180/unit_factor and pred_values[i] <= 70/unit_factor) or (ref_values[i] <= 70/unit_factor and pred_values[i] >= 180/unit_factor):
            zone[4] += 1    #Zone E

        elif ((ref_values[i] >= 70/unit_factor and ref_values[i] <= 290/unit_factor) and pred_values[i] >= ref_values[i] + 110/unit_factor) or ((ref_values[i] >= 130/unit_factor and ref_values[i] <= 180/unit_factor) and (pred_values[i] <= (7/5)*ref_values[i] - 182/unit_factor)):
            zone[2] += 1    #Zone C
        elif (ref_values[i] >= 240/unit_factor and (pred_values[i] >= 70/unit_factor and pred_values[i] <= 180/unit_factor)) or (ref_values[i] <= 175/3/unit_factor and pred_values[i] <= 180/unit_factor and pred_values[i] >= 70/unit_factor) or ((ref_values[i] >= 175/3/unit_factor and ref_values[i] <= 70/unit_factor) and pred_values[i] >= (6/5)*ref_values[i]):
            zone[3] += 1    #Zone D
        else:
            zone[1] += 1    #Zone B
    return fig, zone/len(ref_values)

def consensus_error_grid(ref_values,pred_values,title_string,unit, enable_PEG_20_40, enable_PEG_summary = False):
    
    
    start=time.time()
    
    font = {'family': 'Times New Roman',
    'weight': 'normal',
    'size': 12,
    }
    # fig = plt.figure(figsize=(6,6))
    if unit == 'mg/dL':
        unit_factor = 1
    elif unit == 'mmol/L':
        unit_factor = 18
    

    poly_context_A = {'type': 'MULTIPOLYGON',
        'coordinates': [[[[0, 0], [0, 50/unit_factor], [30/unit_factor, 50/unit_factor], [140/unit_factor, 170/unit_factor],[280/unit_factor,380/unit_factor],[430/unit_factor,550/unit_factor],[550/unit_factor,550/unit_factor],[550/unit_factor,450/unit_factor],[385/unit_factor,300/unit_factor],[170/unit_factor,145/unit_factor],[50/unit_factor,30/unit_factor],[50/unit_factor,0]]]]}
    poly_shape_A = shapely.geometry.shape(poly_context_A)

    poly_context_B = {'type': 'MULTIPOLYGON',
        'coordinates': [[[[0, 0], [0, 60/unit_factor], [30/unit_factor, 60/unit_factor], [50/unit_factor, 80/unit_factor],[70/unit_factor,110/unit_factor],[260/unit_factor,550/unit_factor],[550/unit_factor,550/unit_factor],[550/unit_factor,250/unit_factor],[260/unit_factor,130/unit_factor],[120/unit_factor,30/unit_factor],[120/unit_factor,0]]]]}
    poly_shape_B = shapely.geometry.shape(poly_context_B)

    poly_context_C = {'type': 'MULTIPOLYGON',
        'coordinates': [[[[0, 0], [0, 100/unit_factor], [25/unit_factor, 100/unit_factor], [50/unit_factor, 125/unit_factor],[80/unit_factor,215/unit_factor],[125/unit_factor,550/unit_factor],[550/unit_factor,550/unit_factor],[550/unit_factor,150/unit_factor],[250/unit_factor,40/unit_factor],[250/unit_factor,0]]]]}
    poly_shape_C = shapely.geometry.shape(poly_context_C)

    poly_context_D = {'type': 'MULTIPOLYGON',
        'coordinates': [[[[0, 0], [35/unit_factor, 155/unit_factor], [50/unit_factor, 550/unit_factor], [550/unit_factor, 550/unit_factor],[550/unit_factor,0]]]]}
    poly_shape_D = shapely.geometry.shape(poly_context_D)

    poly_context_E = {'type': 'MULTIPOLYGON',
        'coordinates': [[[[0, 0], [0, 550/unit_factor], [550/unit_factor, 550/unit_factor],[550/unit_factor,0]]]]}
    poly_shape_E = shapely.geometry.shape(poly_context_E)

    x_clarkson_A = np.array([[0,30],[30,140],[140,280],[280,430],[50,50],[50,170],[170,385],[385,550]])/unit_factor
    y_clarkson_A = np.array([[50,50],[50,170],[170,380],[380,550],[0,30],[30,145],[145,300],[300,450]])/unit_factor

    x_clarkson_B = np.array([[0,30],[30,50],[50,70],[70,260],[120,120],[120,260],[260,550]])/unit_factor
    y_clarkson_B = np.array([[60,60],[60,80],[80,110],[110,550],[0,30],[30,130],[130,250]])/unit_factor

    x_clarkson_C = np.array([[0,25],[25,50],[50,80],[80,125],[250,250],[250,550]])/unit_factor
    y_clarkson_C = np.array([[100,100],[100,125],[125,215],[215,550],[0,40],[40,150]])/unit_factor

    x_clarkson_D = np.array([[0,35],[35,50]])/unit_factor
    y_clarkson_D = np.array([[150,155],[155,550]])/unit_factor
    
    x_clarkson_20_u = [0,440]
    y_clarkson_20_u = [0,550]
    x_clarkson_40_u = [0,330]
    y_clarkson_40_u = [0,550]

    x_clarkson_20_l = [0,550]
    y_clarkson_20_l = [0,440]
    x_clarkson_40_l = [0,550]
    y_clarkson_40_l = [0,330]

    if enable_PEG_20_40:
        plt.plot(x_clarkson_20_u, y_clarkson_20_u, linestyle='--', color='g')
        plt.plot(x_clarkson_40_u, y_clarkson_40_u, linestyle='--', color='r')
        plt.plot(x_clarkson_20_l, y_clarkson_20_l, linestyle='--', color='g')
        plt.plot(x_clarkson_40_l, y_clarkson_40_l, linestyle='--', color='r')
    
    plt.title(title_string, font)
    plt.xlabel("Reference Concentration (" + unit + ')', font)
    plt.ylabel("Predicted Concentration (" + unit + ')', font)

    plt.plot([0,550],[0,550],'k:') 
    for i in range (len(x_clarkson_A)):
        plt.plot(x_clarkson_A[i],y_clarkson_A[i],c='black') 

    for i in range (len(x_clarkson_B)):
        plt.plot(x_clarkson_B[i],y_clarkson_B[i],c='black') 

    for i in range (len(x_clarkson_C)):
        plt.plot(x_clarkson_C[i],y_clarkson_C[i],c='black') 

    for i in range (len(x_clarkson_D)):
        plt.plot(x_clarkson_D[i],y_clarkson_D[i],c='black') 

    ref_values = ref_values.reshape(-1,)
    pred_values = pred_values.reshape(-1,)

    # print('程序运行时间为: %s Seconds'%(time.time()-start))
    
    zone = [0] * 5
    
    if enable_PEG_summary:
        for i in range(len(ref_values)):
            point = shapely.geometry.Point(ref_values[i], pred_values[i])
            if poly_shape_A.intersects(point):
                zone[0] += 1     
                plt.scatter(ref_values[i], pred_values[i],marker='x',s = 10,c = 'green')
            elif poly_shape_B.intersects(point):
                zone[1] += 1
                plt.scatter(ref_values[i], pred_values[i],marker='x',s = 10,c = 'orange')            
            elif poly_shape_C.intersects(point):
                zone[2] += 1
                plt.scatter(ref_values[i], pred_values[i],marker='x',s = 10,c = 'red')      
            elif poly_shape_D.intersects(point):
                zone[3] += 1
                plt.scatter(ref_values[i], pred_values[i],marker='x',s = 10,c = 'red')  
            elif poly_shape_E.intersects(point):
                zone[4] += 1
                plt.scatter(ref_values[i], pred_values[i],marker='x',s = 10,c = 'red')          
    else:    
        plt.scatter(ref_values, pred_values,marker='x',s = 10,c = 'black')    
   

    # print('程序运行时间为: %s Seconds'%(time.time()-start))         

    plt.text(350/unit_factor,400/unit_factor,'A',size = 12, font = font)
    plt.text(400/unit_factor,350/unit_factor,'A',size = 12, font = font)
    plt.text(440/unit_factor,60/unit_factor,'D',size = 12, font = font)
    plt.text(10/unit_factor,440/unit_factor,'E',size = 12, font = font)
    plt.text(440/unit_factor,150/unit_factor,'C',size = 12, font = font)
    plt.text(150/unit_factor,440/unit_factor,'C',size = 12, font = font)
    plt.text(60/unit_factor,440/unit_factor,'D',size = 12, font = font)
    plt.text(260/unit_factor,430/unit_factor,'B',size = 12, font = font)
    plt.text(430/unit_factor,260/unit_factor,'B',size = 12, font = font)

    plt.xticks(np.arange(0, 550/unit_factor, step=int(50/unit_factor)))
    plt.yticks(np.arange(0, 550/unit_factor, step=int(50/unit_factor)))
    plt.xlim(0,550/unit_factor)
    plt.ylim(0,550/unit_factor)
    return zone   
    # return fig,zone

# def error_rate_count(ref_values,pred_values):
#     zone = [0] * 10
#     error_rate = abs(ref_values - pred_values)/ref_values * 100
#     for values in error_rate:
#         for i in range(10):
#             if values <= 50 - i*5:
#                 zone[i] += 1
#     return zone

def error_rate_count(ref_values, pred_values, enable_lower_detail=False):
    # 初始化错误率区间列表
    zone = [0] * 10
    
    if enable_lower_detail:
        # 筛选出 ref_values < 100/18 和 ref_values >= 100/18 的值
        mask_lower = ref_values < 100 / 18
        mask_upper = ref_values >= 100 / 18

        # 处理 ref_values < 100/18 的情况
        filtered_ref_values_lower = ref_values[mask_lower]
        filtered_pred_values_lower = pred_values[mask_lower]
        lower_detail_count = 0
        
        if len(filtered_ref_values_lower) > 0:
            lower_detail_count = np.sum(np.abs(filtered_ref_values_lower - filtered_pred_values_lower) < 15 / 18)

        # 处理 ref_values >= 100/18 的情况
        filtered_ref_values_upper = ref_values[mask_upper]
        filtered_pred_values_upper = pred_values[mask_upper]
        upper_detail_count = 0
        
        if len(filtered_ref_values_upper) > 0:
            upper_detail_count = np.sum(np.abs(filtered_ref_values_upper - filtered_pred_values_upper) / filtered_ref_values_upper * 100 < 15)

        # 计算总的符合条件的数量和总的样本数量
        total_count = lower_detail_count + upper_detail_count
        
        # 转换为百分比
        zone[7] = total_count

    else:
        # 计算错误率
        error_rate = np.abs(ref_values - pred_values) / ref_values * 100
        
        # 统计错误率分布
        for values in error_rate:
            for i in range(10):
                if values <= 50 - i * 5:
                    zone[i] += 1
                    
    return zone