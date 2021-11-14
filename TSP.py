"""
Eduardo Saldana
6612626
TSP Problem
Python 3
Visual Studio Code
"""

# Imports
from math import ceil, sqrt
import random
import matplotlib.pyplot as plt
import tkinter as tk

# Creating parameters that will be used later
TOURNAMENT_SIZE = None
GENERATION_SIZE = None
NUM_OF_GENERATIONS = None
CROSSOVER_RATE = None
MUTATION_RATE = None

BEST_ROUTE = None

# Receiving and validating inputs
while True:
    try:
        TOURNAMENT_SIZE = int(input("Choose a size for the tournament (2-5 recommended): "))
        if TOURNAMENT_SIZE<1:
            continue
        break
    except:
        pass
while True:
    try:
        GENERATION_SIZE = int(input("Choose a size for each generation (30-100 is recommended): "))
        if GENERATION_SIZE<1:
            continue
        break
    except:
        pass
while True:
    try:
        NUM_OF_GENERATIONS = int(input("Choose number of generations (100+ is recommended): "))
        if NUM_OF_GENERATIONS<1:
            continue
        break
    except:
        pass
while True:
    try:
        CROSSOVER_RATE = float(input("Crossover rate (0-1): "))
        if CROSSOVER_RATE<0 or CROSSOVER_RATE>1:
            continue
        break
    except:
        pass
while True:
    try:
        MUTATION_RATE = float(input("Mutation Rate (0-1): "))
        if MUTATION_RATE<0 or MUTATION_RATE>1:
            continue
        break
    except:
        pass
while True:
    try:
        seed = float(input("Random Seed: "))
        break
    except:
        pass
random.seed(seed)

MULTIPLE_ELITES = True
NUMBER_ELITES = ceil(GENERATION_SIZE/10)

if MULTIPLE_ELITES==False:
    NUMBER_ELITES = 1



# Creating variables for plot and trackign population
max = -999999999
min = 999999999
mean = None
median = None
stddev = 0

population = []
newpop = []    
xaxis = []
ybest = []
yaverage = []


# Opens the file and reads it as a string
data = open("22locations.txt","r")
text = data.read()
numbers = []
counter = 0

# This appends the numbers from the text into an list
# and append the city number as an int
for x in text.split():
    try:
        if counter%3==0:
            numbers.append(int(x))
        else:
            numbers.append(float(x))
        counter = counter+1
    except:
        pass

counter = 0

cities = []

# Appens the cities into an list
for x in numbers:
    if counter%3==0:
        cities.append(numbers[counter])
    counter = counter + 1

distances = []

#Appends the distances into a list
for x in range(len(numbers)):
    if x%3==1:
        distances.append([numbers[x],numbers[x+1]])

# Calculates euclidean distance by squaring the sum of the difference of x2,x1 and y2,y1
def euclidean(num1, num2):
    cord1 = distances[num1-1]
    cord2 = distances[num2-1]
    length = sqrt(pow((cord2[0]-cord1[0]),2)+pow((cord2[1]-cord1[1]),2))
    return length



# Creates the first generation by generating random orders
def initialPop():
    counter = 0
    while GENERATION_SIZE>counter:
        population.append(random.sample(cities,len(cities)))
        population[counter].append(population[counter][0])
        counter = counter+1

# Calcualtes the fitness of a given chromosome by getting its total distance
# Then takes the inverse so the higher the fitness the better
# Gives an absurdly bad fitness score to invalid chromosomes that dont end at the original city
def fitness(order):
    valid = is_valid_chromosome(order)
    if order[0] != order[len(order)-1] or valid==False:
        return 0
    total_distance = 0
    for x in range(len(order)-1):
        total_distance = total_distance + euclidean(order[x], order[x+1])
    total_distance = 10/total_distance
    return total_distance

# Returns total distance for a given chromosome
def distance(order):
    valid = is_valid_chromosome(order)
    if order[0] != order[len(order)-1] or valid==False:
        return 999999999999999
    total_distance = 0
    for x in range(len(order)-1):
        total_distance = total_distance + euclidean(order[x], order[x+1])
    return total_distance

# Goes through all the options and returns the one with the highest fitness
def tournament(options):
    winner = None
    record = -1
    for x in options:
        if fitness(x)>record:
            record = fitness(x)
            winner = x
    return winner


def crossover_bitmask(parent1, parent2):
    # Creates the childs of the new generation
    child1 = []
    child2 = []
    for x in range(len(cities)+1):
        child1.append(None)
        child2.append(None)
    bitmask = []
    zero_and_one = [0,1]
    # Creates bitmask
    for x in range(len(cities)+1):
        bitmask.append(random.choice(zero_and_one))
    bitmask[0] = 1
    bitmask[len(bitmask)-1] = 1
    # Goes thorugh the bitmask and if the bitmask is 1
    # then each child gets the same number at its parents
    for x in range(len(bitmask)):
        if bitmask[x]==1:
            child1[x] = parent1[x]
            child2[x] = parent2[x]
    
    # Gathers information of what numbers each child is missing based on the order of the second parent
    missing_elements_child1 = []
    missing_elements_child2 = []
    for x in range(len(bitmask)-1):
        if child1.count(parent2[x]) == 0:
            missing_elements_child1.append(parent2[x])
        if child2.count(parent1[x]) == 0:
            missing_elements_child2.append(parent1[x])

    # Fills in the blanks with the numbers of the second parent
    for x in range(len(bitmask)):
        if child1[x]==None:
            child1[x] = missing_elements_child1[0]
            missing_elements_child1.pop(0)
        if child2[x]==None:
            child2[x] = missing_elements_child2[0]
            missing_elements_child2.pop(0)
        

    return child1, child2
    
def two_point_crossover(parent1, parent2):
    # Selects two random numbers which will be the indexes to chose the crossovers
    number1, number2 = random.sample(range(1,len(cities)), 2)

    child1 = []
    child2 = []

    child1_missing = []
    child2_missing = []

    repeated1 = []
    repeated2 = []

    # Orders the indexes by size
    if number2<number1:
        number2, number1 = number1, number2

    # Appens however numbers as the first index with the original parents
    for x in range(number1):
        child1.append(parent1[x])
        child2.append(parent2[x])

    # Appends the opposite parents ordered number as the amount between the two indexes
    for x in range(number2-number1):
        child1.append(parent2[number1+x])
        child2.append(parent1[number1+x])

    # Appens however numbers as the remaining spaces with the original parents
    for x in range(len(cities)-number2+1):
        child1.append(parent1[x+number2])
        child2.append(parent2[x+number2])
    
    # Takes count of what cities is each child missing
    for x in range(len(cities)):
        if child1.count(x+1) == 0:
            child1_missing.append(x+1)
        if child2.count(x+1) == 0:
            child2_missing.append(x+1)

    # Takes count of what cities are repeated in each child
    for x in range(len(cities)):
        if (child1.count(x+1)==2 and (x+1)!=child1[0]) or child1.count(x+1)==3:
            repeated1.append(x+1)
        if (child2.count(x+1)==2 and (x+1)!=child2[0]) or child2.count(x+1)==3:
            repeated2.append(x+1)

    # Replaces repeated cities with whatever cities the child was missing
    for x in child1_missing:
        current = repeated1[0]
        replace = child1.index(current)
        if replace==0:
            replace = child1[1:].index(current)
            replace = replace+1
        child1[replace] = x
        repeated1.pop(0)

    # Replaces repeated cities with whatever cities the child was missing           
    for x in child2_missing:
        current = repeated2[0]
        replace = child2.index(current)
        if replace==0:
            replace = child2[1:].index(current)
            replace = replace+1
        child2[replace] = x
        repeated2.pop(0)
    
    return child1, child2


def is_valid_chromosome(testing):
    # Checks to see if all the numbers are present in the chromosome once and only once with exception to the first and last city
    special_number = testing[0]
    for x in range(len(testing)-1):
        if testing.count(x+1)!=1 and special_number!=x+1:
            return False
    return True
    
def mutation(parent):
    times = random.randint(1, int(round(len(cities)/10)))
    for x in range(times):
        # Picks two random indexes 0-21 that will be swapped
        pick1, pick2 = random.sample(range(0, len(cities)), 2)
        value1 = parent[pick1]
        value2 = parent[pick2]
        # If either index is 0 then both the first and last will be swapped to the other number
        if pick1==0:
            parent[pick2] = value1
            parent[0] = value2
            parent[-1] = value2
            if x==times-1:
                return parent
        
        if pick2==0:
            parent[pick1] = value2
            parent[0] = value1
            parent[-1] = value1
            if x==times-1:
                return parent

        # If neither index is 0 then the two elemtns are simply swapped
        parent[pick1] = value2
        parent[pick2] = value1
        if x==times-1:
                return parent
    
# Creates and writes into document
f = open("Data.txt","w+") 
f.write("Crossover Method: 2 Point Crossover \n")
f.write("Tournament Size: "+str(TOURNAMENT_SIZE) +"\n")
f.write("Generation Size: "+str(GENERATION_SIZE) +"\n")
f.write("Number of generations: "+str(NUM_OF_GENERATIONS) +"\n")
f.write("Crossover Rate: "+str(CROSSOVER_RATE*100) +"% \n")
f.write("Mutation Rate: "+str(MUTATION_RATE*100) +"% \n")
f.write("Number of Elites: "+str(NUMBER_ELITES) +"\n")
f.write("Random Seed: "+str(seed) +"\n")
f.write("\n\n\n")




initialPop()




# Genetic algoritm
def GA():
    global BEST_ROUTE, max, min, mean, stddev, median, population, newpop
    for generation in range(NUM_OF_GENERATIONS):
        
        # Calculates mean
        sum_all = 0
        for x in range(len(population)):
            sum_all = sum_all + distance(population[x])
        
        sum_all = sum_all/len(population)

        yaverage.append(sum_all)

        # Regardless of wether its one or multiple elite chromosomes all of them will be stored in a list and attached back to the populaiton
        # at the end of each loop, the elites are revaluated at the beginning of each loop
        ELITISM = []
        for x in range(NUMBER_ELITES):
            best = tournament(population)
            ELITISM.append(best)
            if x==0:
                best_fitness = distance(best)
            population.pop(population.index(best))

        # Collects data for graph
        xaxis.append(generation+1)
        ybest.append(best_fitness)

        print("Generation "+str(generation+1)+" best chromosome has a length of: "+str(best_fitness))

        # Writes data into text file
        f.write("Generation "+str(generation+1)+" best chromosome has a length of "+str(best_fitness)+"\n")
        f.write("Generation "+str(generation+1)+" chromosomes have an average length of "+str(sum_all)+"\n")
        f.write("\n\n")

        # Applies tournament selection to choose which parents make it
        for x in range(GENERATION_SIZE):
            competitors = []
            # Random chromosomes are taken and the one with the best fitness scores is attached to the new population
            for y in range(TOURNAMENT_SIZE):
                competitors.append(random.choice(population))
            newpop.append(tournament(competitors))
        
        # Replaces the current population with the new generation and rests the new population list variable
        population = newpop
        newpop = []

        # Applies Crossover/Mutation
        for y in range(int(GENERATION_SIZE/2)):
            # Generates the indexes to use for crossover in case the crossover does happen
            indexes = random.sample(range(0, len(population)),2)
            indexes = sorted(indexes)
            cross_chance = random.uniform(0,1)
            if cross_chance<CROSSOVER_RATE:
                """
                HERE YOU CAN CHAGE THE CROSSOVER METHOD, choose between crossover_bitmask (UOX with bitmask) and two_point_crossover (2-point crossover)
                """
                child1, child2 = crossover_bitmask(population[indexes[0]], population[indexes[1]])
                newpop.append(child1)
                newpop.append(child2)
                population.pop(indexes[0])
                population.pop(indexes[1]-1)
            # If the crossover does not occur both aprents just pass normally into the next generation
            else:
                newpop.append(population[indexes[0]])
                newpop.append(population[indexes[1]])
                population.pop(indexes[0])
                population.pop(indexes[1]-1)

        # Replaces the current population with the new generation and rests the new population list variable
        population = newpop
        newpop = []

        # Gives every child in the new population (except elites) and chance to mutate
        for z in range(len(population)):
            mutation_chance = random.uniform(0,1)
            if mutation_chance<MUTATION_RATE:
                population[z] = mutation(population[z])
            
        
        

        # Appends the best chromosome to the pool
        for x in ELITISM:
            population.append(x)
    
    # Saves the best route after each generation
    if MULTIPLE_ELITES==False:
        print(ELITISM)
        BEST_ROUTE = ELITISM
    else:
        ultimate_best = tournament(ELITISM)
        print(ultimate_best)
        BEST_ROUTE = ultimate_best
    
    # Tracks min and max route
    for x in range(len(population)):
        current = distance(population[x])
        if current>max:
            max = current
        if current<min:
            min = current

    # Calculates mean, median and standard deviation, then writes it into the text file
    mean = sum_all

    median = distance(population[int(len(population)/2)])

    for x in range(len(population)):
        stddev = stddev + ((distance(population[x])-mean)**2)

    stddev = stddev/len(population)
    stddev = stddev**0.5
    
    f.write("Max Value: "+str(max) +"\n")
    f.write("Min Value: "+str(min) +"\n")
    f.write("Median Value: "+str(median) +"\n")
    f.write("Standard Deviation Value: "+str(stddev) +"\n")


GA()
f.close()

# Creates graph
plt.title("Routes throughout the generations")
plt.plot(xaxis, ybest, label="Best Current Route (Elite)")
plt.plot(xaxis, yaverage, label="Generation Average Route Length")
plt.ylabel("Route Length")
plt.xlabel("Generation")
plt.legend()
plt.show()

root = tk.Tk()
canvas = tk.Canvas(root, width=1500, height=1500)
canvas.grid()

def circle(self, x, y, r):
    return self.create_oval(x-r, y-r, x+r, y+r, fill="black")
tk.Canvas.create_circle = circle

# Creates circles in Visualization        
for x in range(len(distances)):
    canvas.create_circle(distances[x][0]*15, distances[x][1]*15, 2) 

root.title("Visualization")

# Creates lines in visualization
for x in range(len(BEST_ROUTE)-1):
    now = BEST_ROUTE[x]
    next = BEST_ROUTE[x+1]
    canvas.create_line(distances[now-1][0]*15, distances[now-1][1]*15, distances[next-1][0]*15, distances[next-1][1]*15, width=1, fill='green')

root.mainloop()



