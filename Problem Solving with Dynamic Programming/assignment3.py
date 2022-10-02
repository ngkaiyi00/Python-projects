"""
Name:Ng Kai Yi
Student id:32156944
Assignment 3

"""

# =================================================================
# Question 01
# =================================================================

def floyd_warshall(travel_days):

    """
    This is a function to find the shortest days we need to spend to travel from each city to every other cities.
    if there's no road to travel from one city to another,it will remain as -1.

    post-condition:It will return a matrix which will have the shortest days we need travel from each city to every
    other cities if there's no road to travel from one city to another,it will remain as -1

    :param travel_days:
    A list of list,travel_days[x][y] will contain either:
    • A positive integer number indicating the number of days you need to spend to travel on
    the direct road from city x to city y.
    • -1, to indicate that there is no direct road for you to travel from city x to city y.

    :return:It will return a matrix which will have the shortest days we need to travel from each city to every
    other cities if there's no road to travel from one city to another,it will remain as -1

    Time complexity:O(N^3),N is the length of travel_days which is the number of cities.

    Space complexity:
    Input:O(N^2),N is the length of travel_days
    Aux:O(1)

    """
    count_vertex = len(travel_days)

    for k in range(count_vertex):
        for i in range(count_vertex):
            for j in range(count_vertex):
                if travel_days[i][j] == -1:  # If there's no directed road between city i and j
                    if travel_days[i][k] > 0 and travel_days[k][j] > 0:  # Check if there's a route
                        travel_days[i][j] = travel_days[i][k] + travel_days[k][j]  # Assign it for the first time
                else:
                    if travel_days[i][k] > 0 and travel_days[k][j] > 0:
                        # Get the shorter days spent to travel from city i to j
                        travel_days[i][j] = min(travel_days[i][j], travel_days[i][k] + travel_days[k][j])

    return travel_days


def best_revenue(revenue, travel_days, start):
    """
    This is a function to find the maximum possible revenue,started from the city "start".At first,I used the
    floyd_warshall function created to find the shortest days we need to travel from each city to every
    other cities.My memo here is to record the maximum possible revenue which can be made from day 0 to day i at city j.
    For example,memo[i][j] will have the maximum possible revenue which can be made from day 0 to day i at city j.If its
    on day 0,only the starting city will have the revenue.

    post-condition: memo[i][j] will have the maximum possible revenue which can be made from day 0 to day i at city j.
                    It will return an integer which is the maximum possible revenue

    :param revenue:
    revenue is a list of lists. All interior lists are length n. Each
    interior list represents a different day. revenue[z][x] is the revenue that you would make if
    you work in city x on day z.

    :param travel_days:
    travel_days is a list of lists. travel_days[x][y]will contain either:
    • A positive integer number indicating the number of days you need to spend to travel on
    the direct road from city x to city y.
    • -1, to indicate that there is no direct road for you to travel from city x to city y

    :param start: start denote the city started in on day 0.

    :return:It return an integer which is the maximum possible revenue.

    Time complexity:
    Worst case:
    The time complexity of floyd_warshall used is O(n^3),n is the length of travel_days,which is
    number of cities.The outermost for loop is looping through days,and the inner-loop is looping through cities
    and the innermost is also looping through cities,so the complexity is O(n^2*d),so the overall complexity is
    O(n^3 + n^2*d) which is O(n^2(n+d)

    Best case: Best case = Worst case

    Space complexity:
    O(n^2+nd),n is the length of travel_days,which is number of cities,d is the length of revenue,which is the
    number of days.
    Aux:O(nd),since we will have a memo,which a list of list of length d,and the each list inside the list will have
    the length of n.

    """
    travel_days = floyd_warshall(travel_days)

    memo = [None] * (len(revenue))  # Initialise memo
    for i in range(len(revenue)):  # Looping through Days,starting from day 0
        memo[i] = []
        for j in range(len(travel_days)):  # Looping through Cities,starting from city 0
            
            if i == 0:  # On the first day
                if j == start:
                    # Only append revenue to the starting city because it is impossible to gain revenue on cities other
                    # than starting city on the first day.
                    memo[i].append(revenue[i][j])
                else:
                    memo[i].append(0)

            else:   # Starting from day 1 to last day
                
                if 0 in memo[i-1]:  # If there is still 0 on the previous day

                    # Check if it is possible to travel from starting city to city j,and if the days spent is short
                    # enough,my way to check is see if i - travel_days[start][j] >= 0.                   
                    if travel_days[start][j] > 0 and i - travel_days[start][j] >= 0:
                        memo[i].append(revenue[i][j] + memo[i-1][j])
                    
                    # If the city is the starting city,then we get the revenue on this day + 
                    # the maximum revenue can be made at this city on the previous day.
                    elif start == j: 
                        memo[i].append(revenue[i][j] + memo[i - 1][j])

                    else:
                        memo[i].append(0)
                
                else:
                    # The maximum revenue can be made at this city on the previous day + 
                    # the revenue will be made staying there on day i.
                    memo[i].append(revenue[i][j] + memo[i-1][j])

                for k in range(len(travel_days)):  # Then,loop through each city
                    if travel_days[k][j] > 0:  # If it is possible to travel from city k to city j
                        max_day = travel_days[k][j]  # Days spent to travel
                        if i - max_day - 1 >= 0 and memo[i - max_day - 1][k] != 0:  # If the days spent is short enough
                            # memo[i - max_day - 1][k] is the maximum profit at city k before we travel to j
                            # Find the combination which give the greatest revenue so that we can maximize the revenue
                            # at city j on day i.
                            memo[i][j] = max(memo[i][j], memo[i - max_day - 1][k] + revenue[i][j])

    # Get the last column,which is last day
    result = memo[-1]
    # Sorting
    result.sort()
    # Return the maximum revenue
    return result[-1]


# =================================================================
# Question 02
# =================================================================


def hero(attacks):

    """
    This is a function to return a list of Master X attacks that result in maximum possible clones defeated.
    Firstly,I sort the attacks by the ending days of the attacks.My memo is a list of list with length of len(attacks),
    for example the memo[x][0] store the maximum number of clones can be  defeated if i include attack x,
    then memo[x][1] store the index of the attack which can be included if we include attack x,so that we can backtrack
    by using the value of memo[x][1]. The index is -1 means that no previous attacks other than itself can be included.

    I loop through each attack,then for each attack,I loop through its previous attacks,and use the corresponding
    value stored in the memory to find the possible combination that gives the greatest number of clones to be defeated
    and store the values in the memory for the maximum possible clones can be defeated inclusive of current attack.

    Lastly,I do the backtracking and append the attacks to the list called result and return it.

    Post-condition:It will return a list of list of Master X attacks that result in maximum possible clones defeated.

    :param attacks: attacks is a non-empty list of N attacks,where each attack is a list of 4 items [m, s, e, c]
    :return: Return a list of Master X attacks which results in max possible clones defeated.

    Time complexity:
    Worst case:The worst case is that for each attack,we need to loop from the start of the previous attacks all the way
    to the current attack,so its O(N^2),where N is the number of attacks.

    Best case:The best case is that for each attack,we can terminate early after the first loop,means that the first
    attack will have the end date which is equal to the start date of every other attacks,in this case,the complexity
    become O(N+N) = O(N).So the complexity of sorting is more dominant which is O(NlogN),N is the number of attacks,
    so the best case complexity for this function is O(NlogN).

    Space complexity:
    Input:O(4N),which is O(N),N is the number of attacks
    Aux:O(N) as the memo created is a matrix of N * 2.

    """

    # Base case,if there's only one attack,this is the only possible attack which results in max possible clones
    # defeated.
    if len(attacks) == 1:
        return attacks

    # sort attacks based on column 2, which is e, ending days of the attacks.
    attacks = sorted(attacks, key=lambda x: x[2])

    # Initialise memo
    memo = [None] * len(attacks)
    result = []  # For backtracking

    # My memo is a list of list with length of len(attacks),for example the memo[x][0] store the maximum number of
    # clones can be  defeated if i include attack x, then memo[x][1] store the index of the attack which can be included
    # if we include attack x,so that we can backtrack by using the value of memo[x][1]. The index is -1 if no previous
    # attacks other than itself can be included.

    for i in range(len(attacks)):  # Loop through attacks
        index = -1
        maximum = attacks[i][3]
        # When i == 0,definitely wont have other attacks can be included,so memo[i][0] = -1
        if i == 0:
            memo[i] = [attacks[i][3], -1]
        else:
            memo[i] = [attacks[i][3], -1]

            # loop through the previous attacks
            for j in range(i):
                # If the previous attack has the end date smaller than the start date of the current attack
                if attacks[j][2] < attacks[i][1]:
                    # Then we include the attack which will give the greatest number of clones defeated
                    if memo[j][0] + memo[i][0] > maximum:
                        maximum = memo[j][0] + memo[i][0]
                        index = j
                else:
                    # Terminate early once we find the end date greater than the start date of the current attack since
                    # it is sorted,then store the values into the memo
                    memo[i][0], memo[i][1] = maximum, index
                    break

            # store the values into the memo
            memo[i][0], memo[i][1] = maximum, index

    # Perform backtracking to find all the attacks to be included to get the maximum clones defeated.

    # In memo,find the list with greatest number of clones,so that we know where to backtrack from
    to_track = max(memo, key=lambda x: x[0])

    # Get the index of it and append it into the result list.
    max_index = memo.index(max(memo, key=lambda x: x[0]))
    result.append(attacks[max_index])

    # The index is the index of the next attack we are tracking if its != -1
    index = to_track[1]

    while index != -1:  # If index != -1,means there are still attacks to be included,so we continue to backtrack
        result.append(attacks[index])
        to_track = memo[to_track[1]]
        index = to_track[1]
    return result






