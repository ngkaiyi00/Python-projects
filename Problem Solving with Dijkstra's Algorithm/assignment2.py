import sys

"""
Name:Ng Kai Yi
Student id:32156944
Assignment 2

"""

# =================================================================
# Question 01
# =================================================================


def get_median(num_list, start_of_sublist, end_of_sublist):

    """
    This function is to get the median of the num_list[start_of_sublist:end_of_sublist+1],which will be used
    in the function get med_of_meds to find the median of 5 elements each time.

    Precondition:The num_list is not empty,end_of_sublist - start_of_sublist < 4
    Post-condition:It will return the median of the num_list[start_of_sublist:end_of_sublist+1]

    :param num_list:A list of numbers
    :param start_of_sublist:An index which specify the starting point of the sublist
    :param end_of_sublist:An index which specify the ending point of the sublist
    :return:the median of the num_list[start_of_sublist:end_of_sublist+1]

    Time Complexity:
    Best:
    Since this function will sort a fixed length of list,which is 5,so each sort will take constant time.Since we will
    sort n/5 fixed length of sublist,O(1)*O(N) so its O(N)

    Worst:
    Since this function will sort a fixed length of list,which is 5,so each sort will take constant time.Since we will
    sort n/5 fixed length of sublist,O(1)*O(N) so its O(N)

    Space Complexity:
    Input:O(N) N is the length of the num_list.
    Aux:O(1) since we will create a list of fixed length of 5 so its constant.

    """
    lst = []
    for i in range(start_of_sublist, end_of_sublist+1):
        lst.append(num_list[i])
    lst.sort()

    return lst[(end_of_sublist+1 - start_of_sublist)//2]


def get_med_of_meds(lst, start_of_lst, end_of_lst):

    """
    This function is to get the median of medians.Firstly,it will check if the total number of elements can be fully
    divided into n/5 groups.If can,it will find the median of each group and append into a median list.If cannot,it will
    be divided into n//5 groups and one more group with n%5 element and find the median of each group and append into a
    median list.Then it will check if the length of sublist == 1,if its 1,means only 1 group left,then return the median
    of that group.

    Precondition:The lst is not empty and start_of_lst != end_of_lst
    Post-condition:It will return the median of medians of lst[start_of_lst:end_of:lst+1]

    :param lst:A list of numbers
    :param start_of_lst:An index which specify the starting point of the list/sublist
    :param end_of_lst:An index which specify the ending point of the list/sublist
    :return:Return median of medians.

    Time complexity:
    Best:
    O(N) N is the length of the list.Since we are sorting fixed length of list,so its O(1),we are doing it
    n/5 times so its O(1) * O(N) so O(N)
    Worst:
    Best case = Worst Case.O(N) N is the length of the list.Since we are sorting fixed length of list,so its O(1),
    we are doing it n/5 times so its O(1) * O(N) so O(N)

    Space complexity:
    Input:O(N),N is the length of the lst
    Aux:O(N),median list depends on the size of the lst,N is the length of lst.

    """

    total_elements = end_of_lst - start_of_lst + 1
    medians = []
    num_of_sublist = 0

    # If elements can be divided fully into n/5 groups with length 5
    if total_elements % 5 == 0:
        for x in range(total_elements//5):
            medians.append(get_median(lst, start_of_lst + 5*num_of_sublist, start_of_lst + 5*num_of_sublist+4))
            num_of_sublist += 1

    # Elements need to be divided into n/5 groups with length 5 + one group with length n%5
    elif total_elements % 5 != 0:
        remainder = total_elements % 5
        for x in range(total_elements//5):
            medians.append(get_median(lst, start_of_lst + 5*num_of_sublist, start_of_lst + 5*num_of_sublist+4))
            num_of_sublist += 1

        medians.append(get_median(lst, start_of_lst + 5*num_of_sublist, start_of_lst + 5*num_of_sublist + remainder-1))
        num_of_sublist += 1

    if num_of_sublist == 1:
        med_of_meds = medians[0]

    else:
        med_of_meds = get_med_of_meds(medians, 0, num_of_sublist-1)

    return med_of_meds


def partition(lst, start_of_lst, end_of_lst, med_of_meds):
    """
    This function is to partition the lst with the med_of_meds as pivot.The items on the left of the pivot will be less
    than or equal to the pivot,the items on the right of the pivot will be greater than the pivot.

    Precondition:The lst is not empty.
    Post-condition:It will return the position of the med_of_meds in such a order that the items on the left of the
    pivot will be less than or equal to the pivot,the items on the right of the pivot will be greater than the pivot.

    :param lst: A lst of numbers
    :param start_of_lst:An index which specify the starting point of the list/sublist
    :param end_of_lst:An index which specify the ending point of the list/sublist
    :param med_of_meds: Median of medians
    :return: Return the position of median of medians after partitioning

    Time complexity:
    Best:O(N),Best case = Worst case,because it will loop through the entire lst to do the comparison and partitioning
    Worst:O(N),Best case = Worst case,because it will loop through the entire lst to do the comparison and partitioning

    Space complexity:
    Input:O(N),N = length of the lst
    Aux:O(1)

    """

    # Find median of medians
    for i in range(start_of_lst, end_of_lst+1):
        if lst[i] == med_of_meds:
            lst[end_of_lst], lst[i] = lst[i], lst[end_of_lst]
            break

    # Make median of medians as the pivot
    pivot = lst[end_of_lst]
    x = start_of_lst

    for y in range(start_of_lst, end_of_lst):
        if lst[y] <= pivot:
            lst[y], lst[x] = lst[x], lst[y]
            x += 1

    lst[x], lst[end_of_lst] = lst[end_of_lst], lst[x]
    return x


def quick_select_med_of_meds(lst, start_of_lst, end_of_lst, k):
    """
    This function is to do the quick select by using the pivot found by using the median of medians algorithm.
    Firstly,we get median of medians by calling the method get_med_of_meds.Then,partition the lst by using the median
    found as pivot.Then,we get the position and the rank of median of medians.Rank indicates that it is k th smallest
    number in the lst.Then we compare with the k,if its equal to k,we return the median of medians,if the rank of
    median of medians is greater than k,we recurse to the left of the lst,if the rank of median of medians is
    smaller than k,we recurse to the right to the lst.

    Precondition:The lst is not empty
    Post-condition:It will return the true median of the lst.

    :param lst: A list of number
    :param start_of_lst:An index which specify the starting point of the list/sublist
    :param end_of_lst:An index which specify the ending point of the list/sublist
    :param k: k indicates the kth smallest item in the list.It is the rank of the true median at this case.
    :return: It will return the true median of the lst.

    Time complexity:
    Best:
    The best case will have the complexity of O(N) as well since the best case complexity of get_med_of_meds is O(N),
    the base case time complexity of partition is O(N) and they will be called at least once.
    Worst:
    O(N),N = length of the lst,the time complexity of get_med_of_meds is O(N),the time complexity of partition is O(N).
    T(N/5) is the recursive call to find the median_of_medians,then use it to partition,we can make sure that
    3((N/5)/2-2) elements will be lesser or greater than median of medians.So the next recursive call for
    quick_select_med_of_meds will have the size of the lst of 7n/10+6.So,T(N)=T(N/5)+T(N-3((N/5)/2-2))+O(N)=O(N)


    Space complexity:
    Input:O(N),N is the length of lst
    Aux:O(N),N is the length of lst

    """
    # Get median of medians
    med_of_meds = get_med_of_meds(lst, start_of_lst, end_of_lst)

    # Partitioning using median of medians
    pos_of_med_of_meds = partition(lst, start_of_lst, end_of_lst, med_of_meds)

    rank_of_med_of_meds = pos_of_med_of_meds - start_of_lst + 1

    if rank_of_med_of_meds == k:
        return lst[pos_of_med_of_meds]

    # Recurse to the left
    elif rank_of_med_of_meds > k:
        return quick_select_med_of_meds(lst, start_of_lst, pos_of_med_of_meds-1, k)

    # Recurse to the right
    else:
        return quick_select_med_of_meds(lst, pos_of_med_of_meds+1, end_of_lst, k-rank_of_med_of_meds)


def ideal_place(relevant):
    """
    This is the function will output a single pair of coordinates x and y such that the sum of the manhattan distance
    of this point to all the relevant points is minimal.Firstly,it will compute the median of x-coordinate,followed by
    computing the median of y-coordinate.

    Precondition:The relevant is not empty
    Post-condition:It will return a single pair of coordinates x and y such that the sum of the manhattan distance
    of this point to all the relevant points is minimal.

    :param relevant:Relevant is a list that contains the coordinates of the n relevant points.
    :return:It will return a single pair of coordinates x and y such that the sum of the manhattan distance
    of this point to all the relevant points is minimal.

    Time complexity:
    Best:O(N),since quick_select_med_of_meds has the best case complexity of O(N)
    Worst:O(N),since quick_select_med_of_meds has the worst case complexity of O(N)

    Space complexity:
    Input:O(N),N is the length of relevant
    Aux:O(N),N is the length of relevant
    """

    size = len(relevant)
    k = (size // 2) + 1
    lst = []

    # Create a list which contain all the x-axis
    i = 0
    for x in range(size):
        lst.append(relevant[x][i])

    x_pos = quick_select_med_of_meds(lst, 0, size-1, k)

    i = 1
    lst = []

    # Create a list which contain all the y-axis

    for y in range(size):
        lst.append(relevant[y][i])

    y_pos = quick_select_med_of_meds(lst, 0, size-1, k)

    return [x_pos, y_pos]



#############################################
# Question 2
#############################################


class Heap:
    """
    Citation: Brendon Taylor,Heap implemented using an array.
    https://edstem.org/au/courses/5239/lessons/11939/slides/86298/solution

    This class's code is from FIT1008 unit,Week 12 Workshop Malaysia,the original code is max_heap.I modified it and
    added some methods.

    """
    def __init__(self, size):
        """
        Initialize the min heap.
        precondition:size is not 0
        post-condition:An array of length (size+1) is created.

        :param size: A number which specify the size of the min heap.

        Time Complexity:
        Best:O(N),N is the size
        Worst:O(N),N is the size

        Space Complexity:
        Input:O(1)
        Aux:O(N),N is the size
        """
        self.length = 0
        self.the_array = [None for i in range(size+1)]
        self.the_array[0] = [sys.maxsize*-1, -1]
        self.root = 0

    def __len__(self):
        """
        To get the length of the min heap
        Post-condition:It will return the length of the min heap

        :return: Return the length of the min heap

        Time Complexity:
        Best:O(1)
        Worst:O(1)

        Space Complexity:
        Input:O(1)
        Aux:O(1)

        """
        return self.length

    def is_empty(self):
        """
        This is to check if the min heap is empty.

        post-condition:It will return True if the min heap is empty,return False otherwise
        :return: It will return True if the min heap is empty,return False otherwise

        Time Complexity:
        Best:O(1)
        Worst:O(1)

        Space Complexity:
        Input:O(1)
        Aux:O(1)

        """
        return self.length == 1

    def is_full(self):
        """
        This is to check if the min heap is full.

        post-condition:It will return True if the min heap is full,return False otherwise
        :return: It will return True if the min heap is full,return False otherwise

        Time Complexity:
        Best:O(1)
        Worst:O(1)

        Space Complexity:
        Input:O(1)
        Aux:O(1)

        """
        return self.length + 1 == len(self.the_array)

    def rise(self, k: int):
        """
        Rise element at index k to its correct position
        :precondition: 1 <= k <= self.length
        :post-condition:The element at index k will be greater than its child node,but smaller than its parent node.

        Time Complexity:
        Best:O(1),its when the item[0] is smaller than its parent node's distance,while loop will terminate.
        Worst:O(LogN),N is self.length,the length of the min heap,the item is the smallest out of all the nodes,it will
        move up all the way to the root.

        Space Complexity:
        Input:O(1)
        Aux:O(1)

        """
        item = self.the_array[k]
        while k > 1 and item[0] < self.the_array[k // 2][0]:
            self.the_array[k] = self.the_array[k // 2]
            k = k // 2
        self.the_array[k] = item

    def add(self, key, data):
        """
        Swaps elements while rising
        post-condition:A node will be added to the min heap and will rise to its correct position.

        Time Complexity:
        Best:O(1),the best case of rise() is O(1)
        Worst:O(LogN),the worst case of rise() is O(LogN),N is the length of the min heap.

        Space Complexity:
        Input:O(1)
        Aux:O(1)

        """
        element = [key, data]
        if self.is_full():
            raise IndexError

        self.length += 1
        self.the_array[self.length] = element
        self.rise(self.length)

    def smallest_child(self, k: int) -> int:
        """
        Returns the index of k's child with smallest value.
        :pre: 1 <= k <= self.length // 2
        :post-condition:It will return the smallest child of a node

        Time complexity:
        Best:O(1)
        Worst:O(1)

        Space complexity:
        Input:O(1)
        Aux:O(1)
        """

        if 2 * k == self.length or \
                self.the_array[2 * k][0] < self.the_array[2 * k + 1][0]:
            return 2 * k
        else:
            return 2 * k + 1

    def sink(self, k: int) -> None:
        """
        Make the element at index k sink to the correct position.
            :pre: 1 <= k <= self.length
            :post-condition:The element at index k will be greater than its child node,but smaller than its parent node.

        Time Complexity:
        Best:O(1),its when the item[0] is smaller than its parent node's distance,while loop will terminate.
        Worst:O(LogN),N is self.length,the length of the min heap,the item is the greatest out of all the nodes,it will
        move down all the way to the bottom of the node.

        Space Complexity:
        Input:O(1)
        Aux:O(1)

        """
        item = self.the_array[k]

        while 2 * k <= self.length:
            min_child = self.smallest_child(k)
            if self.the_array[min_child][0] >= item[0]:
                break
            self.the_array[k] = self.the_array[min_child]
            k = min_child

        self.the_array[k] = item

    def get_min(self) -> int:
        """ Remove (and return) the minimum element from the heap.
        post-condition:It will return the minimum element based on the key value from the min heap.

        Time complexity:
        Best:O(1),the best case for sink() is O(1)
        Worst:O(LogN),the worst case for sink() is O(LogN),N is the length of the min heap.

        Space Complexity:
        Input:O(1)
        Aux:O(1)

        """

        if self.length == 0:
            raise IndexError

        min_elt = self.the_array[1]
        self.length -= 1
        if self.length > 0:
            self.the_array[1] = self.the_array[self.length+1]
            self.sink(1)

        return min_elt

    def decrease_key(self, v, distance):
        """
        This is to decrease the distance(key) of the node,then perform up-heap to move the node up to its correct
        position.

        precondition: 0 <= v <= self.length,distance >= 0
        post-condition:The node with the data "v",will have its key changed to "distance" and moved up to correct
        position.

        :param v: v is the vertex id,which is data
        :param distance: distance is the key

        Time complexity:
        Best:O(1),It happens when after the distance is decreased,it is still greater than its parent node.
        Worst:O(LogN),It happens when after the distance is decreased,it become the smallest element,so it will
        move up all the way to the root.

        Space Complexity:
        Input:O(1)
        Aux:O(1)

        """

        for i in range(self.length):
            if self.the_array[i+1][1] == v:
                self.the_array[i+1][0] = distance
                self.rise(i+1)


class Vertex:
    def __init__(self, vertex_id):
        """
        The initialisation of the vertex
        precondition:vertex_id >= 0

        :param vertex_id: Number which is greater than 0

        Time complexity:
        Best:O(1)
        Worst:O(1)

        Space complexity:
        Input:O(1)
        Aux:O(1)

        """
        self.vertex_id = vertex_id
        self.edges = []
        self.discovered = False
        self.visited = False
        self.distance = 0
        self.previous = None

    def __str__(self):
        """
        This is a function to show the information of the vertex id with its edges.
        :return: It will return a string contain the information of the vertex id with its edges.

        Time complexity:
        Best: O(N),N is the length of the edges list
        Worst: O(N),N is the length of the edges list

        Space
        complexity:
        Input: O(1)
        Aux: O(1)

        """

        return_str = ""
        return_str += str(self.vertex_id) + ":"
        for i in range(len(self.edges)):
            return_str += str([self.edges[i].u.vertex_id, self.edges[i].v.vertex_id,self.edges[i].w])
        return return_str


class Edge:
    def __init__(self,u, v, w):
        """
        The initialisation of Edge object.
        precondition: w >= 0

        :param u: u is the source vertex
        :param v: v is the destination vertex
        :param w: w is the weight of the edge connecting u and v

        Time complexity:
        Best: O(N),N is the length of the edges list
        Worst: O(N),N is the length of the edges list

        Space
        complexity:
        Input: O(1)
        Aux: O(1)

        """
        self.u = u
        self.v = v
        self.w = w


class RoadGraph:
    def __init__(self, roads):

        """
        Initialisation of the RoadGraph
        precondition: roads is not empty

        :param roads:A list of roads,and the roads are represented as a list of tuples (u,v,w)

        Time complexity:
        Best: O(N),N is the length of the edges list
        Worst: O(N),N is the length of the edges list

        Space
        complexity:
        Input: O(N),N is the length of the roads
        Aux: O(N),N is the length of the roads,list is created is store vertices.

        """

        self.roads = roads
        size = len(roads)
        temp_size = size * 2
        self.temp_vertices = [None] * temp_size
        self.vertices = []

        # Loop through the roads,add the vertices into the temp_vertices list if they are not in the list.
        for i in range(size):
            if roads[i][0] not in self.temp_vertices:
                self.temp_vertices[roads[i][0]] = Vertex(roads[i][0])
            if roads[i][1] not in self.temp_vertices:
                self.temp_vertices[roads[i][1]] = Vertex(roads[i][1])

        # To remove the None in temp_vertices,we create vertices list to store the vertices.
        for i in range(temp_size):
            if self.temp_vertices[i] is not None:
                self.vertices.append(self.temp_vertices[i])

        # Add all the edges into the corresponding vertex.
        for vertex in range(len(self.vertices)):
            for j in range(size):
                if roads[j][0] == vertex:
                    u = self.vertices[roads[j][0]]
                    v = self.vertices[roads[j][1]]
                    w = roads[j][2]
                    self.vertices[vertex].edges.append(Edge(u,v,w))

    # Print out the RoadGraph,it will print each vertex with its edges in the RoadGraph.

    def __str__(self):
        """
        It will print each vertex with its edges in the RoadGraph.

        Time Complexity:
        Best = Worst = O(N),N is the length of the self.vertices.

        Space complexity:
        Input:O(1)
        Aux:O(1)
        """
        for vertex in self.vertices:
            print(vertex)
        return ""

    def dijkstra(self, source):

        """
        Dijkstra algorithm,given the source,it will find the shortest distance to every vertex from the source.

        precondition: source is one of the elements in self.vertices.
        post-condition:
        It will return a list of distance,indicates the shortest distance from source to each vertex.
        For example,if dijkstra(0) is called,and when the output is for example [0,1,2,3,4],
        the shortest distance from vertex 0 to vertex 1 is 1,to vertex 2 is 2,to vertex 3 is 3 and to vertex 4 is 4,
        the index in the list is correspond to the vertex id of the vertex.

        :param source: Source is the vertex id,given the source,it will find the shortest distance to every
        vertex from the source.

        :return:
        It will return a list of distance,indicates the shortest distance from source to each vertex.
        For example,if dijkstra(0) is called,and when the output is for example [0,1,2,3,4],
        the shortest distance from vertex 0 to vertex 1 is 1,to vertex 2 is 2,to vertex 3 is 3 and to vertex 4 is 4,
        the index in the list is correspond to the vertex id of the vertex.

        Time Complexity:
        Best:Best case = Worst case.
        Worst:The worst case complexity is O(ElogV) where E is the number of edges,and V is the number of vertices.
        Each edge will be visited once,so its O(E),then updating and upheap the min heap is O(logV),and while loop
        execute O(V) times so the total cost is O(E*Q.decrease_key() + V*Q.get_min().So the total cost is
        O(E*logV + V*logV).The graph must be connected,so E dominates V,hence the total cost = O(ElogV)


        Space Complexity:
        Input:O(1)
        Aux:O(V),V is the number of vertices.

        """
        ver = len(self.vertices)
        discovered = Heap(ver)  # Create a min heap
        distance = [None] * ver
        self.vertices[source].distance = 0
        discovered.add(0, self.vertices[source])  # Add the source with key 0 into the min heap.

        while discovered.length > 0:
            ver_distance = discovered.get_min()  # Get the minimum element in the min heap
            u = ver_distance[1]
            u.visited = True
            u.discovered = True
            distance[u.vertex_id] = ver_distance[0]
            for edge in u.edges:
                v = edge.v
                if not v.discovered:  # If v is not discovered
                    v.distance = u.distance + edge.w
                    v.discovered = True
                    discovered.add(v.distance, v)
                    v.previous = u
                else:
                    if not v.visited:  # If the distance of v is not finalised yet
                        if v.distance > u.distance + edge.w:
                            v.distance = u.distance + edge.w
                            discovered.decrease_key(v, u.distance + edge.w)  # Update the key of v in min heap.
                            v.discovered = True
                            v.previous = u  # update the previous vertex of v

        return distance

    def shortest_path(self, source, destination):
        """
        Back-tracking to get the shortest path from source to the destination.

        :param source: Source is the source of the path
        :param destination: Destination is the destination of the path
        :return: It will return a list which is the shortest path from source to destination

        precondition:
        It must perform dijkstra algorithm before calling this method,if we want to set the value of source as k,
        we must perform dijkstra(k) before we do so.

        post-condition: It will return a list which is the shortest path from source to destination

        Time complexity:
        Best:O(1),the loop will terminate if a source.previous is None.
        Worst:O(V),V is the number of vertices.

        Space complexity:
        Input:O(1)
        Aux:O(1)

        """
        if source == destination:
            return [destination]
        lst = []
        ver_destination = self.vertices[destination]
        destination_id = self.vertices[destination].vertex_id

        while ver_destination.previous is not None:
            lst.insert(0, ver_destination.previous.vertex_id)
            ver_destination = ver_destination.previous

        if len(lst) > 0:
            lst.append(destination_id)
            return lst
        else:
            return None

    def routing(self, start, end, chores_location):
        """
        This function will find the shortest path from start to end passing through at least one of the
        chores_location,it will return None if there's no such path exists.I duplicate the original graph
        and pre-process it by changing the direction of all its edges.Then I call dijkstra from end at the pre-processed
        graph to find the shortest distance from end to every other vertices.Then i got the shortest distance from
        start to all other vertices and from end to all other vertices.Then,I loop through them to get the minimum sum
        of distances from start -> chores_location(dist_from_start[chores_location] and
        from end -> chores_location(dist_from_end[chores_location]).After getting the know the minimum sum of the
        distance,I get to know passing through which chore location will give me the minimum distance.Then I find the
        shortest path from start -> that chore location then get the shortest path from end -> chore location and
        reversed the path.After that,combine them to get the final shortest path.

        :param start: start is a non-negative integer that represents the starting location of the journey
        :param end: end is a non-negative integer that represents the ending location of your journey
        :param chores_location:is a non-empty list of non-negative integers that stores all of the location where the
        chores could be performed.

        :return:Return the shortest route from start to end passing through one of the chores_location,return None if
        there's no such path exists.

        Time complexity:
        Best:
        O(ElogV),because the best case of dijkstra is O(ElogV),we are calling it 2 times here,so its
        2*O(ElogV),still O(ElogV),the shortest path function has the best case complexity of O(1),so its still
        O(ElogV).

        Worst:O(ElogV),because the worst case of dijkstra is O(ElogV),we are calling it 2 times here,so its
        2*O(ElogV),still O(ElogV),the shortest path function has the worst case complexity of O(V),so its still
        O(ElogV).

        Space complexity:
        Input:O(N),N is the length of the chores location.
        Aux:O(V),V is the number of vertices,I have 2 list to store the shortest path before combining them.

        """
        try:
            # Call dijkstra from start
            dist_from_start = self.dijkstra(start)
            minimum_dist = sys.maxsize
            minimum_dist_index = None
            roads = self.roads

            #  Reverse the direction of all the edges.
            for i in range(len(roads)):
                roads[i] = list(roads[i])
                roads[i][0], roads[i][1] = roads[i][1], roads[i][0]
                roads[i] = tuple(roads[i])

            #  Create a reversed graph with all the direction of edges reversed.
            reversed_graph = RoadGraph(roads)
            dist_from_end = reversed_graph.dijkstra(end)

            #  If the chore location is at start/end,then we only need to call dijkstra at start.
            for i in range(len(chores_location)):
                if start == chores_location[i] or end == chores_location[i]:
                    final_route = self.shortest_path(start, end)
                    return final_route

            #  Find the minimum sum of distance
            for loc in chores_location:
                # If one of them is None,means there's no such path exists
                if dist_from_start[loc] is not None and dist_from_end[loc] is not None:
                    combined_dist = dist_from_start[loc] + dist_from_end[loc]
                    if combined_dist < minimum_dist:
                        minimum_dist = combined_dist
                        minimum_dist_index = loc  # loc is the chore location which has the minimum sum of distance
                else:
                    pass

            #  Return None if no such path exists.
            if minimum_dist_index is None:
                return None

            #  Get the shortest path from start to the chore location which has the minimum sum of distance
            a_c = self.shortest_path(start, minimum_dist_index)

            #  Get the shortest path from end to the chore location which has the minimum sum of distance
            b_c = reversed_graph.shortest_path(end, minimum_dist_index)

            # Reverse the path from end to the chore location,so that it become a path from chore location to the end.
            b_c.reverse()
            del b_c[0]  # Delete the repeating element before we combine

            # Combine the path from start -> chore location and from chore location -> end
            final_route = a_c + b_c

            return final_route

        except LookupError:
            return None

