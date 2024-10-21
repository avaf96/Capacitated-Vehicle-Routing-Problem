from audioop import add
from cProfile import label
from cmath import inf
import copy
import csv
from errno import ESTALE
import math
from re import T
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
            print("-NotPassed-customer " , customer , " have not been visited." )
            f1 = True
    if not f1:
        print("*PASSED-customer visiting passed." )
    
    f2 = False
    for route in solution:
        custs_demand = 0
        for cust in route:
            custs_demand += demands[cust]
        if custs_demand>capacity:
            print("-NotPassed-route " , route , " violates capacity constraint." )
            f2 = True
    if not f2:
        print("*PASSED-capacity constraint passed." )

    cust_list = []
    for route in solution:
        for c in route:
            cust_list.append(c)
    if len(cust_list)==dimension-1:
        print("*PASSED-no duplicate customer." )
    else:
        print("-NotPassed-some customers are duplicated" )
        print("number of all customers: ", dimension-1)
        print("number of visited customers: " , len(cust_list))




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
               
            q = 0.7
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



# outer_swap function
def outer_swap(route_list):
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
 
    return route_list,total_cost


# migration function
def migration(route_list):
    flag = False
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

        if (demands[cust1] < extra_cap2):     
            flag =True
            route_list[route1].remove(cust1)
            route_list[route2].append(cust1)
        elif (demands[cust2] < extra_cap1):
            flag = True
            route_list[route2].remove(cust2)
            route_list[route1].append(cust2)
        
    route_list = [route for route in route_list if route != []]
        
    total_cost = solution_cost(route_list)
 
    return route_list,total_cost


# inner swap - neighbourhood
def inner_swap(route_list):
    route_list = [route for route in route_list if route != []]
    random_route = random.randint(0,len(route_list)-1)
    cust1_index = random.randint(0,len(route_list[random_route])-1)
    cust2_index = random.randint(0,len(route_list[random_route])-1)
    cust1 = route_list[random_route][cust1_index]    
    cust2 = route_list[random_route][cust2_index]  
    route_list[random_route][cust1_index] = cust2
    route_list[random_route][cust2_index] = cust1
        
    total_cost = solution_cost(route_list)
 
    return route_list,total_cost



# find next customer for a specific customer in a solution
def next_customer(customer,sol):
    for route in sol:
        if customer in route:
            customer_index = route.index(customer)
            if customer_index != len(route)-1:
                cust_neighbor = route[customer_index+1]
                return cust_neighbor
            else:
                return -1


# find previous customer for a specific customer in a solution
def previous_customer(customer,sol):
    for route in sol:
        if customer in route:
            customer_index = route.index(customer)
            if customer_index != 0:
                cust_neighbor = route[customer_index-1]
                return cust_neighbor
            else:
                return -1
    



# Genetic algorithm 
def GA_algorithm():
    exec_time = 0
    start_time = time.time()

    best_sols = []

    pop_size = 50
    pop = {}
    pop_cost = []
    while(pop_size>0):
        current_solution,current_solution_cost = initialSolution()
        pop[round(current_solution_cost,2)] = current_solution
        pop_cost.append(round(current_solution_cost,2))
        pop_size -= 1


    itr = 200
    eval_count = 0
    while itr>0 and exec_time < 30:
        pop_cost.sort()
        pop_cost1 = pop_cost.copy()
        parents_cost = pop_cost1[:6]
        for i in range(6):
            pop_cost1.remove(pop_cost1[i])
        for i in range(8):
            rand_cust = random.choice(pop_cost1)
            parents_cost.append(rand_cust)
            pop_cost1.remove(rand_cust)

        parents = []
        for cost in parents_cost:
            parents.append(pop[cost])

        best_cost = pop_cost[0]
        best_sol = pop[best_cost]
        best_sols.append(best_cost)
        
            
        for i in range(0,len(parents),2):
            p1_sol = parents[i]
            p2_sol = parents[i+1]

            # crossover
            pc = 0.9
            crossover_rand = random.uniform(0,1)
            if crossover_rand <= pc:
                p1 = 0.5
                p2 = 0.5
                probs_num = [1,2]
                probs = [p1,p2]
                
                customers1 = []
                for i in range(1,dimension):
                    customers1.append(i)
                g1 = []
                while(customers1):
                    each_route = []
                    selected_customer = 0
    
                    next_available_custs = find_next_custs(customers1,each_route) 
                    remian_custs_probs = find_next_custs_probs(selected_customer,next_available_custs)

                    while(next_available_custs):
                        prob_method = roulette_Wheel(probs_num,probs)
                        # p1 probability
                        if prob_method==1:
                            if selected_customer==0:
                                routes_list_len = len(g1)
                                if routes_list_len<=len(p1_sol)-1 and p1_sol[routes_list_len][0] in customers1:
                                    selected_customer = p1_sol[routes_list_len][0]
                                else:
                                    selected_customer = random.choice(customers1)
                            else:
                                next_cust = next_customer(last_customer,p1_sol)
                                previous_cust = previous_customer(last_customer,p1_sol)
                                if next_cust in next_available_custs:
                                    selected_customer = next_cust
                                elif previous_cust in next_available_custs:
                                    selected_customer = previous_cust
                                else:    
                                    q = 0.75
                                    q0 = round(random.random(),3)
                                    if q0 < q:
                                        max_prob = min(remian_custs_probs)                      
                                        max_prob_index = remian_custs_probs.index(max_prob)
                                        selected_customer = next_available_custs[max_prob_index] 
                                    else:    
                                        selected_customer = roulette_Wheel(next_available_custs,remian_custs_probs)
                    
                            each_route.append(selected_customer)
                            customers1.remove(selected_customer)    

                        # p2 probability
                        elif prob_method==2:
                            if selected_customer==0:
                                routes_list_len = len(g1)
                                if routes_list_len<=len(p2_sol)-1 and p2_sol[routes_list_len][0] in customers1:
                                    selected_customer = p2_sol[routes_list_len][0]
                                else:
                                    selected_customer = random.choice(customers1)
                            else:
                                next_cust = next_customer(last_customer,p2_sol)
                                previous_cust = previous_customer(last_customer,p2_sol)
                                if next_cust in next_available_custs:
                                    selected_customer = next_cust
                                elif previous_cust in next_available_custs:
                                    selected_customer = previous_cust
                                else:    
                                    q = 0.75
                                    q0 = round(random.random(),3)
                                    if q0 < q:
                                        max_prob = min(remian_custs_probs)                      
                                        max_prob_index = remian_custs_probs.index(max_prob)
                                        selected_customer = next_available_custs[max_prob_index] 
                                    else:    
                                        selected_customer = roulette_Wheel(next_available_custs,remian_custs_probs)

                            each_route.append(selected_customer)
                            customers1.remove(selected_customer)

                        next_available_custs = find_next_custs(customers1,each_route) 
                        remian_custs_probs = find_next_custs_probs(selected_customer,next_available_custs)
                        last_customer = each_route[len(each_route)-1]

                    g1.append(each_route)  


                customers2 = []
                for i in range(1,dimension):
                    customers2.append(i)
                g2 = []
                while(customers2):
                    each_route = []
                    selected_customer = 0
    
                    next_available_custs = find_next_custs(customers2,each_route) 
                    remian_custs_probs = find_next_custs_probs(selected_customer,next_available_custs)

                    while(next_available_custs):
                        prob_method = roulette_Wheel(probs_num,probs)
                        # p1 probability
                        if prob_method==1:
                            if selected_customer==0:
                                routes_list_len = len(g2)
                                if routes_list_len<=len(p1_sol)-1 and p1_sol[routes_list_len][0] in customers2:
                                    selected_customer = p1_sol[routes_list_len][0]
                                else:
                                    selected_customer = random.choice(customers2)
                            else:
                                next_cust = next_customer(last_customer,p1_sol)
                                previous_cust = previous_customer(last_customer,p1_sol)
                                if next_cust in next_available_custs:
                                    selected_customer = next_cust
                                elif previous_cust in next_available_custs:
                                    selected_customer = previous_cust
                                else:    
                                    q = 0.75
                                    q0 = round(random.random(),3)
                                    if q0 < q:
                                        max_prob = min(remian_custs_probs)                      
                                        max_prob_index = remian_custs_probs.index(max_prob)
                                        selected_customer = next_available_custs[max_prob_index] 
                                    else:    
                                        selected_customer = roulette_Wheel(next_available_custs,remian_custs_probs)
                    
                            each_route.append(selected_customer)
                            customers2.remove(selected_customer)    

                        # p2 probability
                        elif prob_method==2:
                            if selected_customer==0:
                                routes_list_len = len(g2)
                                if routes_list_len<=len(p2_sol)-1 and p2_sol[routes_list_len][0] in customers2:
                                    selected_customer = p2_sol[routes_list_len][0]
                                else:
                                    selected_customer = random.choice(customers2)
                            else:
                                next_cust = next_customer(last_customer,p2_sol)
                                previous_cust = previous_customer(last_customer,p2_sol)
                                if next_cust in next_available_custs:
                                    selected_customer = next_cust
                                elif previous_cust in next_available_custs:
                                    selected_customer = previous_cust
                                else:    
                                    q = 0.75
                                    q0 = round(random.random(),3)
                                    if q0 < q:
                                        max_prob = min(remian_custs_probs)                      
                                        max_prob_index = remian_custs_probs.index(max_prob)
                                        selected_customer = next_available_custs[max_prob_index] 
                                    else:    
                                        selected_customer = roulette_Wheel(next_available_custs,remian_custs_probs)

                            each_route.append(selected_customer)
                            customers2.remove(selected_customer)

                        next_available_custs = find_next_custs(customers2,each_route) 
                        remian_custs_probs = find_next_custs_probs(selected_customer,next_available_custs)
                        last_customer = each_route[len(each_route)-1]

                    g2.append(each_route)

            else:
                continue
        

            # mutation
            pm = 0.3
            mutations = [1,2]
            mutation_probs = [0.5,0.5]
            mutation_rand = random.uniform(0,1)
            if mutation_rand <= pm:
                rand_mute1 = roulette_Wheel(mutations,mutation_probs)
                if rand_mute1==1:
                    g1 , g1_cost = outer_swap(g1)
                    eval_count+=1
                elif rand_mute1==2:
                    g1 , g1_cost = inner_swap(g1)
                    eval_count+=1

                rand_mute2 = roulette_Wheel(mutations,mutation_probs)
                if rand_mute2==1:
                    g2 , g2_cost = outer_swap(g2) 
                    eval_count+=1
                elif rand_mute2==2:
                    g2 , g2_cost = inner_swap(g2) 
                    eval_count+=1
            else:
                g1_cost = solution_cost(g1)
                eval_count+=1
                g2_cost = solution_cost(g2)
                eval_count+=1

           
            # add offsprings to population
            pop[round(g1_cost,2)] = g1
            pop[round(g2_cost,2)] = g2
            pop_cost.append(round(g1_cost,2))
            pop_cost.append(round(g2_cost,2))
            
            
        itr -= 1
        end_time = time.time()
        exec_time = int(round(end_time - start_time))/60


    
    # visualRoutes(best_sol)
    # plot_solution_selection_process(best_sols)
    return best_sol,best_cost,eval_count  
 




# -------------------------------3itr run--------------------------------------
# # header = ("instance" , "best" , "worst" , "avg_result" ,"gap" )
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
#     eval_count_list = []
#     for z in range (0,itr):
#         best_solution , best_solution_cost , eval_count = GA_algorithm()
#         result.append(best_solution_cost)
#         res_sol.append(best_solution)
#         eval_count_list.append(eval_count)

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
#     # save_best_solution(res_sol[best_index],path)
#     best_eval_count = eval_count_list[best_index]

#     up = upperBand()
#     gap = round((best-up)/up*100 , 2)


#     print("best: " , round(best , 2))
#     print("worst: " , round(worst ,2))
#     print("avg_res: " , round(avg_res,2))
#     print("gap: " , round(gap ,2))
#     print("eval count: " , best_eval_count)


 
#     # dataa1 = []
#     # da = []
#     # da.append(path)
#     # da.append(round(best ,2))
#     # da.append(round(worst ,2))
#     # da.append(round(avg_res ,2))
#     # da.append(round(gap ,2))
#     # dataa.append(da)
#     # dataa1.append(da)

#     # with open('C:\\Users\\Desktop\\algo\\GA1_result.csv', 'a', encoding='UTF8', newline='') as f:
#     #     writer = csv.writer(f)
#     #     writer.writerows(dataa1)
    

# # with open('C:\\Users\\Desktop\\algo\\GA_result.csv', 'a', encoding='UTF8', newline='') as f:
# #     writer = csv.writer(f)
# #     writer.writerows(dataa)








# # -------------------------------5min run--------------------------------------
header = ("instance" ,  "best_solution_cost" , "gap" , "eval_count" )
with open('C:\\Users\\Desktop\\algo\\GA_result_5min.csv', 'w', encoding='UTF8', newline='') as f:
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
    best_solution , best_solution_cost , eval_count = GA_algorithm()
    end_time = time.time()

    save_best_solution(best_solution,path)
    # feasibility_check(best_solution)

    up = upperBand()
    gap = round((best_solution_cost-up)/up*100 , 2)


    print("best_solution_cost: " , round(best_solution_cost , 2))
    print("gap: " , round(gap ,2))
    print("eval_count: " , eval_count)
    print("execution time: " , int(round(end_time - start_time))/60)
    
 
    dataa1 = []
    da = []
    da.append(path)
    da.append(round(best_solution_cost ,2))
    da.append(round(gap ,2))
    da.append(eval_count)
    dataa.append(da)
    dataa1.append(da)

    with open('C:\\Users\\Desktop\\algo\\GA1_result_5min.csv', 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(dataa1)
    

with open('C:\\Users\\Desktop\\algo\\GA_result_5min.csv', 'a', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(dataa)



    

        
    