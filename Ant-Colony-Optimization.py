from audioop import add
from cProfile import label
from cmath import inf
import copy
import csv
from errno import ESTALE
import math
from re import T
from sys import flags
from traceback import print_tb
from turtle import color
import numpy as np
import os
from scipy import spatial
import matplotlib.pyplot as plt
import random
import time
import pandas as pd


# global variables
dimension = 0
k_min = 0
capacity = 0
points = []
demands = []
costs = np.empty((dimension, dimension))

ro = 0.1
sigma = 0.5
ro_global = 0.01
Q = 500

# read file and  set values
def readFile(addrs):
    lines = []
    with open(addrs, 'r') as f:
        data = f.readlines()
        for line in data:
            line = line.strip('|').split()
            lines.append(line)
        f.close

    file_name = lines[0][2].split("-")
    dimension = int(file_name[1][1: ])
    k_min = int(file_name[2][1: ])
    capacity = int(lines[5][2])

    points = []
    for i in range(0,dimension):
        str_to_int = list(map(int, lines[i+7][1:]))
        points.append(str_to_int)
        

    demands = []
    for i in range(0,dimension):
        str_to_int = list(map(int, lines[i+8+dimension][1:]))
        demands.append(str_to_int[0])

    costs = np.empty((dimension, dimension))
    for i in range(0, dimension):
        for j in range(0, dimension):
            costs[i][j] = "{:.3f}".format(spatial.distance.euclidean(points[i],points[j]))

    return(dimension,k_min,capacity,demands,costs,points)


# initial global variables
def global_variables_initialization(addrs):
    global dimension
    global k_min
    global capacity 
    global demands 
    global costs
    global points
    dimension,k_min,capacity,demands,costs,points = readFile(addrs)


# visual depot and customers
def visualPoints():
    depot_x_cord = points[0][0]
    depot_y_cord = points[0][1]
    customers_x_cord = []
    customers_y_cord = []
    for i in range(1, len(points)):
        customers_x_cord.append(points[i][0])
        customers_y_cord.append(points[i][1])
    plt.scatter(depot_x_cord,depot_y_cord,color=['midnightblue'] , marker = "*" , label='depot' , s=200) 
    plt.scatter(customers_x_cord,customers_y_cord,color=['deepskyblue'] , marker = "." ,label='customers' , s=200) 
    plt.grid(color = 'gray', linestyle = '--', linewidth = 0.3)
    plt.legend(bbox_to_anchor=(1, 1),loc='upper left')
    plt.xlim(-20, 1020)
    plt.ylim(-20, 1020)
    plt.show()


# visual routes
def visualRoutes(routesList):
    depot_x_cord = points[0][0]
    depot_y_cord = points[0][1]
    plt.scatter(depot_x_cord,depot_y_cord,color=['black'] , marker = "*" , s=200) 
    for route in routesList:
        x_cords = []
        y_cords = []
        for point in route:
            x_cords.append(points[point][0])
            y_cords.append(points[point][1])
        route_num = routesList.index(route)
        plt.plot(x_cords, y_cords , marker = "." , label =f'route {route_num}') 
    plt.grid(color = 'gray', linestyle = '--', linewidth = 0.3) 
    plt.legend(bbox_to_anchor=(1, 1),loc='upper left')
    plt.xlim(-20, 1020)
    plt.ylim(-20, 1020)
    plt.show()


# plot solution selection process
def plot_solution_selection_process(best_solutions):
    solutions_cost = best_solutions
    num_of_sols = []
    for i in range(0,len(best_solutions)):
        num_of_sols.append(i+1)
    
    plt.plot(num_of_sols , solutions_cost)
    plt.show()


# retrieve upperband for each file to calculate gap
def upperBand(addrs2 ="C:\\Users\\Desktop\\algo\\BKS.xlsx"): 
    upper_band = 0
    df = pd.read_excel (addrs2)
    df1 = df.get(["n","UB"])
    for i in range(0,30):
        if df1.iloc[i, 0]+1 == dimension:
            upper_band = df1.iloc[i, 1]
    
    return upper_band


# check if a solution is feasible
def feasibility_check(solution):
    solution_length = len(solution)
    if solution_length < k_min:
        print('less than k-min routes have been used.')
    else:
        print('k-min constrait passed')

    customers = []
    for i in range(1,dimension):
        customers.append(i)

    f1 = False
    for customer in customers:
        flag = False
        for route in solution:
            if customer in route:
                flag = True
        if not flag:
            print("customer " , customer , " have not been visited." )
            f1 = True
    if not f1:
        print("customer visiting passed." )
    
    f2 = False
    for route in solution:
        custs_demand = 0
        for cust in route:
            custs_demand += demands[cust]
        if custs_demand>capacity:
            print("route " , route , " violates capacity constraint." )
            f2 = True
    if not f2:
        print("capacity constraint passed." )


# save best solution routes in a text file
def save_best_solution(solution , path):
    sol_route_list = ''
    for i in range(0,len(solution)):
        route_cost = 0
        sol = ''
        for j in range(0,len(solution[i])):
            # route cost 
            if j==0 and j==(len(solution[i])-1):
                route_cost += costs[0][solution[i][j]]
                route_cost += costs[solution[i][j]][0]
            elif j==0:
                route_cost += costs[0][solution[i][j]]
                route_cost += costs[solution[i][j]][solution[i][j+1]]
            elif j==(len(solution[i])-1):
                route_cost += costs[solution[i][j]][0] 
            else:
                route_cost += costs[solution[i][j]][solution[i][j+1]]

            # route
            if j==len(solution[i])-1:
                sol += str(solution[i][j])
            else:
                sol += str(solution[i][j]) + "," 

        # print("route" , i , "==> " , sol )
        sol_route_list += f'route {i} ==> {sol}  -> C={round(route_cost,2)} \n'


    with open('C:\\Users\\Desktop\\algo\\route-list-for-each-file\\'+ path , 'w', encoding='UTF8') as f:
        f.write(sol_route_list)


# calculate costs/distance
def solution_cost(routeList):
    total_cost = 0
    for i in routeList:
        route_cost = 0
        for j in range(len(i)):
            if j==0 and j==(len(i)-1):
                route_cost += costs[0][i[j]]
                route_cost += costs[i[j]][0]
            elif j==0:
                route_cost += costs[0][i[j]]
                route_cost += costs[i[j]][i[j+1]]
            elif j==(len(i)-1):
                route_cost += costs[i[j]][0] 
            else:
                route_cost += costs[i[j]][i[j+1]]

        total_cost += route_cost
    return total_cost


# Russian Roulette Selection
def roulette_Wheel(population,probs):  
    max = sum([i for i in probs])
    selection_probs = [i / max for i in probs]
    selected = np.random.choice(population, p=selection_probs)
    return selected


# find probability of next allowed customers for each route
def find_next_custs_probs(customer,probs,remian_customers):
    remian_custs_probs = []
    if remian_customers:
        for remain_cust in remian_customers:
            remian_custs_probs.append(probs[customer][remain_cust])
    return remian_custs_probs


# find next allowed customers for each route
def find_next_custs(custs,route):
        next_available_custs = []
        route_demand = 0
        for r in route:
            route_demand += demands[r]
        extra_cap = capacity - route_demand

        for cust in custs:
            if demands[cust] <= extra_cap:
                next_available_custs.append(cust)

        return next_available_custs


# evaporation for arcs which are in worst solution 
def evaporation(solution,pheromone_matrix):
    evaporation_amount = 1 - ro 
    for route in solution:
        for cust_index in range(len(route)):
            if cust_index==0 and cust_index == len(route)-1:
                pheromone_matrix[0][route[cust_index]] = pheromone_matrix[0][route[cust_index]] * evaporation_amount
                pheromone_matrix[route[cust_index]][0] = pheromone_matrix[route[cust_index]][0] * evaporation_amount
            elif cust_index==0:
                pheromone_matrix[0][route[cust_index]] = pheromone_matrix[0][route[cust_index]] * evaporation_amount
                pheromone_matrix[route[cust_index]][route[cust_index+1]] = pheromone_matrix[route[cust_index]][route[cust_index+1]] * evaporation_amount
            elif cust_index == len(route)-1:
                pheromone_matrix[route[cust_index]][0] = pheromone_matrix[route[cust_index]][0] * evaporation_amount
            else:
                pheromone_matrix[route[cust_index]][route[cust_index+1]] = pheromone_matrix[route[cust_index]][route[cust_index+1]] * evaporation_amount
    

# global evaporation for all arcs which are in pheromone matrix
def global_evaporation(pheromone_matrix):
    evaporation_amount = 1 - ro_global 
    for i in range(len(pheromone_matrix)):
        for j in range(len(pheromone_matrix)):
            pheromone_matrix[i][j] = pheromone_matrix[i][j] * evaporation_amount


# check pheromone amount if its zero or not. if its zero it will change to 0.2
def check_pheromone_amount(pheromone_matrix):
    for i in range(len(pheromone_matrix)):
        for j in range(len(pheromone_matrix)):
                pheromone_matrix[i][j] = 0.2


# pheromone amount update for arcs which are in best solution 
def pheromone_update(best_sol,best_sol_cost,curren_best_sol_cost,pheromone_matrix):
    # pheromone_amount = (sigma*best_sol_cost/curren_best_sol_cost)
    pheromone_amount = (Q/best_sol_cost) 
    # pheromone_amount = sigma

    for route in best_sol:
        for cust_index in range(len(route)):
            if cust_index==0 and cust_index == len(route)-1:
                pheromone_matrix[0][route[cust_index]] = pheromone_matrix[0][route[cust_index]] + pheromone_amount
                pheromone_matrix[route[cust_index]][0] = pheromone_matrix[route[cust_index]][0] + pheromone_amount
            elif cust_index==0:
                pheromone_matrix[0][route[cust_index]] = pheromone_matrix[0][route[cust_index]] + pheromone_amount
                pheromone_matrix[route[cust_index]][route[cust_index+1]] = pheromone_matrix[route[cust_index]][route[cust_index+1]] + pheromone_amount
            elif cust_index == len(route)-1:
                pheromone_matrix[route[cust_index]][0] = pheromone_matrix[route[cust_index]][0] + pheromone_amount
            else:
                pheromone_matrix[route[cust_index]][route[cust_index+1]] = pheromone_matrix[route[cust_index]][route[cust_index+1]] + pheromone_amount



# ACO algorithm
def ACO():
    local_best_plot = []
    global_best_plot = []
    total_eval = 0

    alfa = 1
    beta = 2 
    
    # customers array
    customers = []
    for i in range(1,dimension):
        customers.append(i)


    # attractiveness matrix - value = 1/distance
    attractiveness = np.empty((dimension, dimension))
    for i in range(0, dimension):
        for j in range(0, dimension):
            if costs[i][j] != 0:
                attractiveness[i][j] = "{:.3f}".format(1/costs[i][j])
            else:
                attractiveness[i][j] = 0
    

    # pheromone matrix - initial value = 1
    pheromone = np.empty((dimension, dimension))
    for i in range(0, dimension):
        for j in range(0, dimension):
            if costs[i][j] != 0:
                pheromone[i][j] = 1
            else:
                pheromone[i][j] = 0


    
    best_sol = []
    best_sol_cost = float('inf')
    exec_time = 0
    start_time = time.time()

    if dimension>=100 and dimension<200:
        itr = 40
    elif dimension>=200 and dimension<400:
        itr = 30
    elif dimension>=400:
        itr = 10

    while itr>0 and exec_time < 20:
        # probability matrix
        probability = np.empty((dimension, dimension))
        for i in range(0, dimension):
            for j in range(0, dimension):
                if i!=j:
                    probability[i][j] = (pow(pheromone[i][j],alfa) * pow(attractiveness[i][j],beta))
                else:
                    probability[i][j] = 0


        current_best_sol = []
        current_best_sol_cost = float('inf')
        current_worst_sol = []
        current_worst_sol_cost = 0

        if dimension>=100 and dimension<200:
            ants_num = 10
        elif dimension>=200 and dimension<400:
            ants_num = 7
        elif dimension>=400:
            ants_num = 3

        while ants_num > 0:
            custs = customers.copy()
            routes_list = []
            
            while(custs):
                selected_customer = 0
                each_route = []
                each_route.append(selected_customer)

                next_available_custs = find_next_custs(custs,each_route) 
                remian_custs_probs = find_next_custs_probs(selected_customer,probability,next_available_custs)

                while(next_available_custs):
                    custs_prob_sum = 0
                    for i in next_available_custs:
                        custs_prob_sum += (pow(pheromone[selected_customer][i],alfa) * pow(attractiveness[selected_customer][i],beta))
                   

                    for prob in remian_custs_probs:
                        if custs_prob_sum==0:
                            prob=0
                        else:
                            prob = prob/custs_prob_sum
                        
                    q = 0.88
                    q0 = round(random.random(),3)
                    if q0 < q:
                        max_prob = max(remian_custs_probs)                      
                        max_prob_index = remian_custs_probs.index(max_prob)
                        selected_customer = next_available_custs[max_prob_index] 

                    else:    
                        selected_customer = roulette_Wheel(next_available_custs,remian_custs_probs)

                    each_route.append(selected_customer)
                    custs.remove(selected_customer)    

                    next_available_custs = find_next_custs(custs,each_route) 
                    remian_custs_probs = find_next_custs_probs(selected_customer,probability,next_available_custs)
                
                each_route.remove(0)
                routes_list.append(each_route)
                

            sol_cost = solution_cost(routes_list)
            total_eval +=1    
            if sol_cost < current_best_sol_cost:
                current_best_sol.clear()
                current_best_sol = copy.deepcopy(routes_list)
                current_best_sol_cost = sol_cost
            local_best_plot.append(current_best_sol_cost)

            if sol_cost > current_worst_sol_cost:
                current_worst_sol.clear()
                current_worst_sol = copy.deepcopy(routes_list)
                current_worst_sol_cost = sol_cost

            ants_num -= 1

        if current_best_sol_cost < best_sol_cost:
            best_sol.clear()
            best_sol = copy.deepcopy(current_best_sol)
            best_sol_cost = current_best_sol_cost
        global_best_plot.append(best_sol_cost)

       
        evaporation(current_worst_sol,pheromone)
        pheromone_update(best_sol,best_sol_cost,current_best_sol_cost,pheromone)
        check_pheromone_amount(pheromone)
        global_evaporation(pheromone)
        
 
        end_time = time.time()
        exec_time = int(round(end_time - start_time))/60
        itr-=1
        

    # plot_solution_selection_process(local_best_plot)
    # plot_solution_selection_process(global_best_plot)
    # visualRoutes(best_sol)
    return best_sol,best_sol_cost,total_eval






    







# # -------------------------------3itr run--------------------------------------
# # filename = "C:\\Users\\Desktop\\algo\\result.csv"
# # header = ("instance" , "best" , "worst" , "avg_result" , "avg_cpu_time" , "best_sol_eval_times" ,"gap" )
# # with open('C:\\Users\\Desktop\\algo\\GA_result.csv', 'w', encoding='UTF8', newline='') as f:
# #     writer = csv.writer(f)
# #     writer.writerow(header)
# # dataa = []
# d = "C:\\Users\\Desktop\\algo\\STD-Ins"
# for path in os.listdir(d):
#     full_path = os.path.join(d, path)
#     global_variables_initialization(full_path)
#     print(path)
#     # visualPoints()

#     itr = 3
#     result = []
#     res_sol = []
#     eval_num = []
#     cpu_time = []
#     for z in range (0,itr):
#         start_time = time.time()
#         solution , sol_cost , total_eval = ACO()
#         result.append(round(sol_cost,2))
#         res_sol.append(solution)
#         eval_num.append(total_eval)
#         cpu_time.append(int(round(time.time()-start_time))/60)

#     best = result[0]
#     worst = result[0]
#     avg_res = result[0]
#     for z in range (1,len(result)):
#         if result[z] < best:
#             best = result[z]
#         if result[z] > worst:
#             worst = result[z]
#         avg_res += result[z]
#     avg_res = avg_res/len(result)

#     avg_cpu_time = 0
#     for z in range (0,itr):
#         avg_cpu_time += cpu_time[z] 
#     avg_cpu_time = avg_cpu_time/itr
         

#     best_index = result.index(best)
#     # save_best_solution(res_sol[best_index],path)

#     best_sol_eval_times = eval_num[best_index]

#     # feasibility_check(full_path,res_sol[best_index])
    

#     up = upperBand()
#     gap = round((best-up)/up*100 , 2)

#     print("best: " , round(best , 2))
#     print("worst: " , round(worst ,2))
#     print("avg_res: " , round(avg_res,2))
#     print("avg cpu time: " , round(avg_cpu_time,2))
#     print("best solution evaluation: " , best_sol_eval_times)
#     print("gap: " , round(gap ,2))
 
#     # dataa1 = []
#     # da = []
#     # da.append(path)
#     # da.append(round(best ,2))
#     # da.append(round(worst ,2))
#     # da.append(round(avg_res ,2))
#     # da.append(round(avg_cpu_time,2))
#     # da.append(best_sol_eval_times)
#     # da.append(round(gap ,2))
#     # dataa.append(da)
#     # dataa1.append(da)

# #     with open('C:\\Users\\Desktop\\algo\\aco1_result.csv', 'a', encoding='UTF8', newline='') as f:
# #         writer = csv.writer(f)
# #         writer.writerows(dataa1)
    

# # with open('C:\\Users\\Desktop\\algo\\aco_result.csv', 'a', encoding='UTF8', newline='') as f:
# #     writer = csv.writer(f)
# #     writer.writerows(dataa)








# # -------------------------------5min run--------------------------------------
# filename = "C:\\Users\\Desktop\\algo\\result.csv"
header = ("instance" ,  "best_solution_cost" , "gap" , "total_evaluation" )
with open('C:\\Users\\Desktop\\algo\\aco_result_5min.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
dataa = []
d = "C:\\Users\\Desktop\\algo\\STD-Ins"
for path in os.listdir(d):
    full_path = os.path.join(d, path)
    global_variables_initialization(full_path)
    print(path)
    # visualPoints()

    start_time = time.time()
    best_sol , best_sol_cost , total_eval = ACO()
    end_time = time.time()

    save_best_solution(best_sol,path)
    # feasibility_check(best_sol)

    up = upperBand()
    gap = round((best_sol_cost-up)/up*100 , 2)

    print("best_solution_cost: " , round(best_sol_cost , 2))
    print("gap: " , round(gap ,2))
    print("total eval: " , total_eval)
    print("execution time: " , int(round(end_time - start_time))/60)
 
    dataa1 = []
    da = []
    da.append(path)
    da.append(round(best_sol_cost ,2))
    da.append(round(gap ,2))
    da.append(total_eval)
    dataa.append(da)
    dataa1.append(da)

    with open('C:\\Users\\Desktop\\algo\\aco1_result_5min.csv', 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(dataa1)
    

with open('C:\\Users\\Desktop\\algo\\aco_result_5min.csv', 'a', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(dataa)



    

        
    