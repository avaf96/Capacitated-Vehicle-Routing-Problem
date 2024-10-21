from audioop import add
from cProfile import label
from cmath import inf
import copy
import csv
from errno import ESTALE
import math
from sys import flags
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


# read file and set values for each instance
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

        sol_route_list += f'route {i} ==> {sol}  -> C={round(route_cost,2)} \n'

    with open('C:\\Users\\Desktop\\algo\\route-list-for-each-file\\'+ path , 'w', encoding='UTF8') as f:
        f.write(sol_route_list)



# find upperband for each instance from excell file 
def upperBand(addrs2 ="C:\\Users\\Desktop\\algo\\BKS.xlsx"):
    upper_band = 0
    df = pd.read_excel (addrs2)
    df1 = df.get(["n","UB"])
    for i in range(0,30):
        if df1.iloc[i, 0]+1 == dimension:
            upper_band = df1.iloc[i, 1]
    
    return upper_band



# visual depot and customers
def visualPoints():
    depot_x_cord = points[0][0]
    depot_y_cord = points[0][1]
    customers_x_cord = []
    customers_y_cord = []
    for i in range(1, len(points)):
        customers_x_cord.append(points[i][0])
        customers_y_cord.append(points[i][1])
    plt.scatter(depot_x_cord,depot_y_cord,color=['midnightblue'] , marker = "*" , label='depot', s=200) 
    plt.scatter(customers_x_cord,customers_y_cord,color=['deepskyblue'] , marker = "." ,label='customers',s=200) 
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
        plt.plot(x_cords, y_cords , marker = "." , label =f'route {route_num}' ) 
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



# # initial solution
# def initialSolution():
#     customers = []
#     for i in range(1,dimension):
#         customers.append(i)

#     route_list =[]
#     total_cost = 0
#     rt = []

#     for i in range(0,k_min):
#         rt = []
#         rand_cust = random.randint(0,len(customers)-1)
#         rt.append(customers[rand_cust])
#         customers.remove(customers[rand_cust])
#         route_list.append(rt)

#     while(customers):
#         rand_cust = random.randint(0,len(customers)-1)
#         rand_route = random.randint(0,len(route_list)-1) 
#         route_list[rand_route].append(customers[rand_cust])
#         customers.remove(customers[rand_cust])

#     ext_cust_list = []
#     for r in route_list:
#         route_demand = 0
#         for c in r:
#             route_demand += demands[c]
#         extra = route_demand - capacity 
#         while extra > 0:
#             rand_cust = random.randint(0,len(r)-1)
#             cu = r[rand_cust] 
#             extra = extra - demands[cu]
#             ext_cust_list.append(cu)
#             r.remove(cu)

#     for ext_cust in ext_cust_list:
#         for route in route_list:
#             route_demand = 0
#             for custo in route:
#                 route_demand += demands[custo]
#             extr = capacity - route_demand
#             if demands[ext_cust] <= extr:
#                 route.append(ext_cust)
#                 ext_cust_list.remove(ext_cust)
#                 break

#     while(ext_cust_list):
#         dem = 0
#         list1 = []
#         for ext_cust in ext_cust_list:
#             if demands[ext_cust] + dem <= capacity:
#                 dem += demands[ext_cust]
#                 list1.append(ext_cust)
#                 ext_cust_list.remove(ext_cust)
#         route_list.append(list1)

#     total_cost = 0
#     for i in route_list:
#         route_cost = 0
#         for j in range(len(i)):
#             if j==0 and j==(len(i)-1):
#                 route_cost += costs[0][i[j]]
#                 route_cost += costs[i[j]][0]
#             elif j==0:
#                 route_cost += costs[0][i[j]]
#                 route_cost += costs[i[j]][i[j+1]]
#             elif j==(len(i)-1):
#                 route_cost += costs[i[j]][0] 
#             else:
#                 route_cost += costs[i[j]][i[j+1]]

#         total_cost += route_cost

#     return route_list,total_cost




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


# find probability of next allowed customers for each route
def find_next_custs_probs(customer,remian_customers):
    remian_custs_probs = []
    if remian_customers:
        for remain_cust in remian_customers:
            remian_custs_probs.append(costs[customer][remain_cust])
    return remian_custs_probs



# initial solution 
def initialSolution():
    customers = []
    for i in range(1,dimension):
        customers.append(i)

    routes_list = []
    while(customers):
        selected_customer = 0
        each_route = []
        each_route.append(selected_customer)

        next_available_custs = find_next_custs(customers,each_route) 
        remian_custs_probs = find_next_custs_probs(selected_customer,next_available_custs)

        while(next_available_custs):
               
            q = 0.8
            q0 = round(random.random(),3)
            if q0 < q:
                max_prob = min(remian_custs_probs)                      
                max_prob_index = remian_custs_probs.index(max_prob)
                selected_customer = next_available_custs[max_prob_index] 

            else:    
                selected_customer = roulette_Wheel(next_available_custs,remian_custs_probs)

            each_route.append(selected_customer)
            customers.remove(selected_customer)    

            next_available_custs = find_next_custs(customers,each_route) 
            remian_custs_probs = find_next_custs_probs(selected_customer,next_available_custs)
        
        each_route.remove(0)
        routes_list.append(each_route)
                
    sol_cost = solution_cost(routes_list)
    
   
    return routes_list,sol_cost

# check if a solution is feasible
def feasibility_check(solution):
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

    cust_list = []
    for route in solution:
        for c in route:
            cust_list.append(c)
    if len(cust_list)==dimension-1:
        print("All customers have been visited." )
    else:
        print("some customers have not been visited" )
        print(dimension-1)
        print(len(cust_list))


# # neighbourhood solutio
# def neighbourhood(routeList):
#     route_list = copy.deepcopy(routeList)
#     total_cost = 0
#     flag = False
#     exchanged_custs = []
#     ct = 0
#     while(flag == False):
#         route1 = random.randint(0,len(route_list)-1)
#         while(not route_list[route1]):
#             route1 = random.randint(0,len(route_list)-1)
#         route2 = random.randint(0,len(route_list)-1)
#         while (route2 == route1) or (not route_list[route2]):
#             route2 = random.randint(0,len(route_list)-1)

#         cust1_index = random.randint(0,len(route_list[route1])-1)
#         cust2_index = random.randint(0,len(route_list[route2])-1)

#         cust1 = route_list[route1][cust1_index]    
#         cust2 = route_list[route2][cust2_index]  

#         cust_demand1 = 0
#         for cust in route_list[route1]:
#             if cust != cust1:
#                 cust_demand1 += demands[cust]
#         extra_cap1 = capacity - cust_demand1

#         cust_demand2 = 0
#         for cust in route_list[route2]:
#             if cust != cust2:
#                 cust_demand2 += demands[cust]
#         extra_cap2 = capacity - cust_demand2

#         if (demands[cust1] < extra_cap2) and (demands[cust2] < extra_cap1):
#             ct += 1
#             if ct==1:
#                 flag = True            
#             exchanged_custs.append((cust1,cust2))
#             route_list[route1].remove(cust1)
#             route_list[route1].append(cust2)
#             route_list[route2].remove(cust2)
#             route_list[route2].append(cust1)
        
#     for route in route_list:  
#         if (not route):
#             route_list.remove(route)
        
#     for route in route_list:  
#         route_cost = 0      
#         for j in range(len(route)):
#             if j==0 and j==(len(route)-1):
#                 route_cost += costs[0][route[j]]
#                 route_cost += costs[route[j]][0]
#             elif j==0:
#                 route_cost += costs[0][route[j]]
#                 route_cost += costs[route[j]][route[j+1]]
#             elif j==(len(route)-1):
#                 route_cost += costs[route[j]][0] 
#             else:
#                 route_cost += costs[route[j]][route[j+1]]

#         total_cost += route_cost
 
#     return route_list,total_cost,exchanged_custs


#  neighbourhood
def neighbourhood(route_list):
    flag = False
    exchanged_custs = []

    while(flag == False):
        route1 = random.randint(0,len(route_list)-1)
        while(not route_list[route1]):
            route1 = random.randint(0,len(route_list)-1)
        route2 = random.randint(0,len(route_list)-1)
        while(not route_list[route2]):
            route2 = random.randint(0,len(route_list)-1)

        cust1_index = random.randint(0,len(route_list[route1])-1)
        cust2_index = random.randint(0,len(route_list[route2])-1)

        cust1 = route_list[route1][cust1_index]    
        cust2 = route_list[route2][cust2_index]  

        cust_demand1 = 0
        for cust in route_list[route1]:
            if cust != cust1:
                cust_demand1 += demands[cust]
        extra_cap1 = capacity - cust_demand1

        cust_demand2 = 0
        for cust in route_list[route2]:
            if cust != cust2:
                cust_demand2 += demands[cust]
        extra_cap2 = capacity - cust_demand2

        if (demands[cust1] < extra_cap2) and (demands[cust2] < extra_cap1):     
            flag =True     
            exchanged_custs.append((cust1,cust2))
            route_list[route1].remove(cust1)
            route_list[route1].append(cust2)
            route_list[route2].remove(cust2)
            route_list[route2].append(cust1)
        
    route_list = [route for route in route_list if route != []]
        
    total_cost = solution_cost(route_list)
 
    return route_list,total_cost,exchanged_custs


# Tabu Search
def TS_algorithm():
    exec_time = 0
    start_time = time.time()

    T_list = [] 
    current_solution , current_solution_cost = initialSolution()
    solution_count = 1
    initial_cost = current_solution_cost
    best_cost = current_solution_cost  
    best_visited = []
    best_visited = copy.deepcopy(current_solution)
    best_solution_process = []
    best_solution_process.append(best_cost)

    itr = 0
    while(itr <=30 and exec_time < 20): 
        
        for _ in range (0,50):
            if len(T_list) >= 8:
                T_list.pop(0)
            
            neighbour_solution , neighbour_solution_cost , exchanged_customers = neighbourhood(current_solution)
            solution_count +=1
            if (neighbour_solution_cost < current_solution_cost):
                flag = False
                if exchanged_customers[0] in T_list: flag = True
            
                if flag == False:  
                    current_solution.clear()
                    current_solution = copy.deepcopy(neighbour_solution)
                    current_solution_cost = neighbour_solution_cost
                    T_list.append(exchanged_customers[0])
                    
        
        #check if current solution is better than best solution or not
        if current_solution_cost < best_cost:
            best_cost = current_solution_cost
            best_visited.clear()
            best_visited = copy.deepcopy(current_solution) 
            best_solution_process.append(best_cost)

        itr +=1

        end_time = time.time()
        exec_time = int(round(end_time - start_time))/60
   

    # visualRoutes(best_visited)
    # plot_solution_selection_process(best_solution_process)
    return best_visited,best_cost,solution_count,initial_cost
    

      
    

# header = ("instance" , "avg_initial_cost", "best" , "worst" , "avg_result" , "avg_cpu_time", "avg_sol_count", "avg_best_sol_count" , "gap")
# with open('C:\\Users\\Desktop\\algo\\TS_result.csv', 'w', encoding='UTF8', newline='') as f:
#     writer = csv.writer(f)
#     writer.writerow(header)
# dataa = []
# d = "C:\\Users\\Desktop\\algo\\STD-Ins"
# for path in os.listdir(d):
#     full_path = os.path.join(d, path)
#     global_variables_initialization(full_path)
#     print(path)
#     # visualPoints()

#     itr = 3
#     result = []
#     cpu_time = []
#     solution_count_list = []
#     best_solution_count_list = []
#     initial_cost_list = []
#     for z in range (0,itr):
#         start_time = time.time()
#         best_solution_cost,solution_eval,initial_cost = TS_algorithm()
#         result.append(best_solution_cost)
#         solution_count_list.append(solution_eval)
#         initial_cost_list.append(initial_cost)
#         cpu_time.append(int(round(time.time()-start_time))/60)
    

#     avg_cpu_time = 0
#     for z in range (0,itr):
#         avg_cpu_time += cpu_time[z] 
#     avg_cpu_time = avg_cpu_time/itr

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

#     best_index = result.index(best)
#     best_sol_eval = solution_count_list[best_index]
#     best_sol_initial_cost = initial_cost_list[best_index]

#     up = upperBand()
#     gap = round((best-up)/up*100 , 2)

#     print("best: " , round(best , 2))
#     print("worst: " , round(worst ,2))
#     print("avg_res: " , round(avg_res,2))
#     print("avg_cpu_time: " , round(avg_cpu_time ,2))
#     print("best_sol_eval: " , round(best_sol_eval ,2))
#     print("best_sol_initial_cost: " , round(best_sol_initial_cost ,2))
#     print("gap: " , round(gap ,2))
 
    
    # da = []
    # da.append(path)
    # da.append(round(best_sol_initial_cost,2))
    # da.append(round(best ,2))
    # da.append(round(worst ,2))
    # da.append(round(avg_res ,2))
    # da.append(round(avg_cpu_time ,2))
    # da.append(round(best_sol_eval ,2))
    # da.append(round(gap ,2))
    # dataa.append(da)


# with open('C:\\Users\\Desktop\\algo\\TS_result.csv', 'a', encoding='UTF8', newline='') as f:
#     writer = csv.writer(f)
#     writer.writerows(dataa)





header = ("instance" , "initial_cost","best_solution_cost" , "solution_eval" ,"gap")
with open('C:\\Users\\Desktop\\algo\\TS1_result.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
dataa = []
d = "C:\\Users\\Desktop\\algo\\STD-Ins"
for path in os.listdir(d):
    full_path = os.path.join(d, path)
    global_variables_initialization(full_path)
    print(path)

    start_time = time.time()
    best_visited_sol,best_solution_cost,solution_eval,initial_cost, = TS_algorithm()
    end_time = time.time()

    up = upperBand()
    gap = round((best_solution_cost-up)/up*100 , 2)

    save_best_solution(best_visited_sol,path)
    # feasibility_check(best_visited_sol)

 
    print("initial_cost: " , round(initial_cost,2))
    print("best_solution_cost: " , round(best_solution_cost,2))
    print("solution_eval: " , round(solution_eval ,2))
    print("execution time: " , int(round(end_time - start_time))/60)
    print("gap: " , round(gap ,2))


    dataa1 = []
    da = []
    da.append(path)
    da.append(round(initial_cost,2))
    da.append(round(best_solution_cost ,2))
    da.append(round(solution_eval ,2))
    da.append(round(gap ,2))
    dataa.append(da)
    dataa1.append(da)
    
    with open('C:\\Users\\Desktop\\algo\\TS1_result.csv', 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(dataa1)

with open('C:\\Users\\Desktop\\algo\\TS2_result.csv', 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(dataa)