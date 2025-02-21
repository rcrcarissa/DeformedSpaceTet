import h5py
import numpy as np
import pickle
import networkx as nx
import copy
from shapely.geometry import Point, Polygon, LinearRing
import sys
import vtk
import argparse


class Tree:
    def __init__(self, root):
        self.root = root
        self.nodes = []

    def addNode(self, node):
        self.nodes.append(node)

    def addBranch(self, subtree):
        self.nodes += subtree.nodes


class Node:
    def __init__(self, data):
        self.parent = None
        self.children = []
        self.data = data  # adjacency list; the node to be eliminated (pivot node); separated tets

    def addChild(self, node):
        self.children.append(node)
        node.parent = self


def appendElementInValue(dict, key, element):
    try:
        dict[key].append(element)
    except:
        dict[key] = [element]
    return dict


def removearray(L, arr):
    idx = 0
    size = len(L)
    while idx < size and not np.array_equal(L[idx], arr):
        idx += 1
    if idx < size:
        L.pop(idx)
        return L
    else:
        raise ValueError('Array not found in list.')


def is_four_points_coplanar(p, q, r, s):
    """
    Args: Coordinates of four points
    Return: True/False
    """
    A = q - p
    B = r - p
    C = s - p
    return np.abs(np.dot(A, np.cross(B, C))) < 1e-7


def node2coor(node, nodes_coor_lower, nodes_coor_upper):
    node_coor_3d = np.zeros(3)
    if node[1] == 'l':
        node_coor_3d[:-1] = nodes_coor_lower[node[0]]
        node_coor_3d[-1] = 0
    else:
        node_coor_3d[:-1] = nodes_coor_upper[node[0]]
        node_coor_3d[-1] = 1
    return node_coor_3d


def sortTriangleNodes(n0, n1, n2):
    '''
    Args:
        n0, n1, n2: Three nodes in a triangle in (node_idx,'layer') format.
    Returns:
        a 3-tuple representing a triangle with ordered nodes
         such that(1) )lower nodes come before upper nodes and (2) lower and upper nodes are sorted
    '''
    nodes = np.array([list(n0), list(n1), list(n2)])
    nodes = nodes[nodes[:, 0].argsort()]
    sorted_nodes_lower = []
    sorted_nodes_upper = []
    for n in nodes:
        if n[1] == 'l':
            sorted_nodes_lower.append((int(n[0]), 'l'))
        else:
            sorted_nodes_upper.append((int(n[0]), 'u'))
    return tuple(sorted_nodes_lower + sorted_nodes_upper)


def exportAdjacencyList(comps_lower, comps_upper, cells_lower, cells_upper, next_nodes):
    '''
    Args:
        indSys: [a list of cells in lower layer, a list of cells in upper layer]
        next_nodes: a list of next nodes
    Returns:
        adjacency_list: a dict with nodes as keys and their adjacency lists as values. Every node is represented by (idx, 'l') or (idx, 'u')
        Separation of real faces and penta-faces is also considered.
        boundary_triangles: all boundary triangles with both upper and lower nodes.
    '''
    node_list = set()
    edge_list = set()
    lower_boundary_edge = set()
    for cell in comps_lower:
        cell = cells_lower[cell]
        node_list.add((cell[0], 'l'))
        node_list.add((cell[1], 'l'))
        node_list.add((cell[2], 'l'))
        sorted_nodes = np.sort(cell)
        edge_list.add(((sorted_nodes[0], 'l'), (sorted_nodes[1], 'l')))
        edge_list.add(((sorted_nodes[0], 'l'), (sorted_nodes[2], 'l')))
        edge_list.add(((sorted_nodes[1], 'l'), (sorted_nodes[2], 'l')))
        edge_list.add(((sorted_nodes[0], 'l'), (next_nodes[sorted_nodes[0]], 'u')))
        edge_list.add(((sorted_nodes[1], 'l'), (next_nodes[sorted_nodes[1]], 'u')))
        edge_list.add(((sorted_nodes[2], 'l'), (next_nodes[sorted_nodes[2]], 'u')))
        if ((sorted_nodes[0], 'l'), (sorted_nodes[1], 'l')) in lower_boundary_edge:
            lower_boundary_edge.remove(((sorted_nodes[0], 'l'), (sorted_nodes[1], 'l')))
        else:
            lower_boundary_edge.add(((sorted_nodes[0], 'l'), (sorted_nodes[1], 'l')))
        if ((sorted_nodes[0], 'l'), (sorted_nodes[2], 'l')) in lower_boundary_edge:
            lower_boundary_edge.remove(((sorted_nodes[0], 'l'), (sorted_nodes[2], 'l')))
        else:
            lower_boundary_edge.add(((sorted_nodes[0], 'l'), (sorted_nodes[2], 'l')))
        if ((sorted_nodes[1], 'l'), (sorted_nodes[2], 'l')) in lower_boundary_edge:
            lower_boundary_edge.remove(((sorted_nodes[1], 'l'), (sorted_nodes[2], 'l')))
        else:
            lower_boundary_edge.add(((sorted_nodes[1], 'l'), (sorted_nodes[2], 'l')))
    for cell in comps_upper:
        cell = cells_upper[cell]
        node_list.add((cell[0], 'u'))
        node_list.add((cell[1], 'u'))
        node_list.add((cell[2], 'u'))
        sorted_nodes = np.sort(cell)
        edge_list.add(((sorted_nodes[0], 'u'), (sorted_nodes[1], 'u')))
        edge_list.add(((sorted_nodes[0], 'u'), (sorted_nodes[2], 'u')))
        edge_list.add(((sorted_nodes[1], 'u'), (sorted_nodes[2], 'u')))
    node_list = sorted(node_list)
    edge_list = sorted(edge_list)
    adjacency_list = {}
    for node in node_list:
        adjacency_list[node] = []
    for edge in edge_list:
        adjacency_list[edge[0]].append(edge[1])
        adjacency_list[edge[1]].append(edge[0])
    inner_nodes = [node for node, adj_list in adjacency_list.items() if 'l' not in [n[1] for n in adj_list]]
    boundary_triangles = []
    for edge in lower_boundary_edge:
        if (next_nodes[edge[0][0]], 'u') in adjacency_list[(next_nodes[edge[1][0]], 'u')]:
            adjacency_list = addEdgeInAdjacencyList((edge[0][0], 'l'), (next_nodes[edge[1][0]], 'u'), adjacency_list)
            boundary_triangles.append(
                sortTriangleNodes((edge[0][0], 'l'), (edge[1][0], 'l'), (next_nodes[edge[1][0]], 'u')))
            boundary_triangles.append(
                sortTriangleNodes((edge[0][0], 'l'), (next_nodes[edge[1][0]], 'u'), (next_nodes[edge[0][0]], 'u')))
        elif next_nodes[edge[0][0]] == next_nodes[edge[1][0]]:
            boundary_triangles.append(((edge[0][0], 'l'), (edge[1][0], 'l'), (next_nodes[edge[0][0]], 'u')))
        else:
            for inner_n in inner_nodes:
                if (next_nodes[edge[0][0]], 'u') in adjacency_list[inner_n] and (next_nodes[edge[1][0]], 'u') in \
                        adjacency_list[inner_n]:
                    adjacency_list = addEdgeInAdjacencyList((edge[0][0], 'l'), inner_n, adjacency_list)
                    adjacency_list = addEdgeInAdjacencyList((edge[1][0], 'l'), inner_n, adjacency_list)
                    inner_nodes.remove(inner_n)
                    boundary_triangles.append(sortTriangleNodes((edge[0][0], 'l'), (edge[1][0], 'l'), inner_n))
                    boundary_triangles.append(
                        sortTriangleNodes((edge[0][0], 'l'), (next_nodes[edge[0][0]], 'u'), inner_n))
                    boundary_triangles.append(
                        sortTriangleNodes((edge[1][0], 'l'), inner_n, (next_nodes[edge[1][0]], 'u')))
                    break
    return adjacency_list, boundary_triangles


def addEdgeInAdjacencyList(node1, node2, adjacency_list):
    adjacency_list[node1].append(node2)
    adjacency_list[node2].append(node1)
    return adjacency_list


def removeNodeInAdjacencyList(node, adjacency_list):
    for n in adjacency_list[node]:
        adjacency_list[n].remove(node)
    del adjacency_list[node]
    return adjacency_list


def orientTet(tet, nodes_coor_lower, nodes_coor_upper):
    tet_n0 = node2coor(tet[0], nodes_coor_lower, nodes_coor_upper)
    tet_n1 = node2coor(tet[1], nodes_coor_lower, nodes_coor_upper)
    tet_n2 = node2coor(tet[2], nodes_coor_lower, nodes_coor_upper)
    tet_n3 = node2coor(tet[3], nodes_coor_lower, nodes_coor_upper)
    p0 = tet_n1 - tet_n0
    p1 = tet_n2 - tet_n0
    p2 = tet_n3 - tet_n0
    sign = np.dot(np.cross(p0, p1), p2)
    if sign > 0:
        return tet
    return [tet[0], tet[2], tet[1], tet[3]]


def removeNode(edges_to_add, tets_to_remove, bound_tris_to_remove, bound_tris_to_add, tree_node, tree, nodes_coor_lower,
               nodes_coor_upper):
    adj_list = copy.deepcopy(tree_node.data['adj_list'])
    boundary_triangles = copy.deepcopy(tree_node.data['boundary_tris'])
    pivot = tree_node.data['pivot']
    for edge in edges_to_add:
        adj_list = addEdgeInAdjacencyList(edge[0], edge[1], adj_list)
    if len(tets_to_remove) > 0:
        adj_list = removeNodeInAdjacencyList(pivot, adj_list)
    for tri in bound_tris_to_remove:
        boundary_triangles.remove(tri)
    for tri in bound_tris_to_add:
        boundary_triangles.append(tri)
    for i in range(len(tets_to_remove)):
        tets_to_remove[i] = orientTet(tets_to_remove[i], nodes_coor_lower, nodes_coor_upper)
    child = Node(
        {'adj_list': adj_list, 'boundary_tris': boundary_triangles, 'pivot': None,
         'separated_tets': tree_node.data['separated_tets'] + tets_to_remove, 'unvisited_children': []})
    tree_node.addChild(child)
    tree.addNode(child)
    if len(tets_to_remove) == 0:
        tree_node.data['unvisited_children'].append(child)
    return child, tree_node, tree


def findSmallestCycle(node1, node2, node_min, isolated_nodes, adjacency_list):
    if node1[1] == 'l':
        node_temp = node1
        node1 = node2
        node2 = node_temp
    if node1 in adjacency_list[node2]:
        return [node_min, node1, node2]
    else:
        adj_nodes1 = set(adjacency_list[node1]) - set(isolated_nodes)
        adj_nodes2 = set(adjacency_list[node2]) - set(isolated_nodes)
        common_node = adj_nodes1.intersection(adj_nodes2)
        if len(common_node) > 0:
            if len(common_node) > 1:
                print("More than one 4-cycles detected between two nodes!")
            else:
                return [node_min, node1, list(common_node)[0], node2]
        else:
            for inter_node in adj_nodes1:
                inter_node_adj_nodes = set(adjacency_list[inter_node]) - set(isolated_nodes)
                common_node = inter_node_adj_nodes.intersection(adj_nodes2)
                if len(common_node) > 0:
                    if len(common_node) > 1:
                        print("More than one 5-cycles detected between two nodes!")
                    else:
                        return [node_min, node1, inter_node, list(common_node)[0], node2]
            return False


def isSegmentInVolume(n0, n1, boundary_triangles, nodes_coor_lower, nodes_coor_upper):
    # take the midlines of boundary triangles
    midlines = []
    for tri in boundary_triangles:
        lower_nodes = [n[0] for n in tri if n[1] == 'l']
        upper_nodes = [n[0] for n in tri if n[1] == 'u']
        if len(lower_nodes) == 1:
            midlines.append(((nodes_coor_lower[lower_nodes[0]] + nodes_coor_upper[upper_nodes[0]]) / 2,
                             (nodes_coor_lower[lower_nodes[0]] + nodes_coor_upper[upper_nodes[1]]) / 2))
        else:
            midlines.append(((nodes_coor_upper[upper_nodes[0]] + nodes_coor_lower[lower_nodes[0]]) / 2,
                             (nodes_coor_upper[upper_nodes[0]] + nodes_coor_lower[lower_nodes[1]]) / 2))

    # sort the midlines of boundary triangles such that they form a cycle
    sorted_polygon = [midlines[0][0], midlines[0][1]]
    current_point = sorted_polygon[-1]
    midlines = midlines[1:]
    while len(midlines) > 1:
        for line in midlines:
            if line[0][0] == current_point[0] and line[0][1] == current_point[1]:
                sorted_polygon.append(line[1])
                current_point = sorted_polygon[-1]
                midlines = removearray(midlines, line)
            elif line[1][0] == current_point[0] and line[1][1] == current_point[1]:
                sorted_polygon.append(line[0])
                current_point = sorted_polygon[-1]
                midlines = removearray(midlines, line)

    # point-in-polygon (PIP) test
    if n0[1] == 'l':
        n0_coor = nodes_coor_lower[n0[0]]
    else:
        n0_coor = nodes_coor_upper[n0[0]]
    if n1[1] == 'l':
        n1_coor = nodes_coor_lower[n1[0]]
    else:
        n1_coor = nodes_coor_upper[n1[0]]
    midpoint = (n0_coor + n1_coor) / 2
    pt = Point(midpoint[0], midpoint[1])
    poly = Polygon(sorted_polygon)
    ring = LinearRing(sorted_polygon)
    return pt.within(poly) or pt.within(ring)


def nodeElimination(adj_list, boundary_tris, nodes_coor_lower, nodes_coor_upper):
    root = Node({'adj_list': adj_list, 'boundary_tris': boundary_tris, 'pivot': None, 'separated_tets': [],
                 'unvisited_children': []})
    tree = Tree(root)
    current_node = root
    split_signal = True
    while current_node.data['adj_list']:
        if not split_signal:
            while True:
                if current_node != tree.root and len(current_node.parent.data['unvisited_children']) == 0:
                    current_node = current_node.parent
                elif current_node != tree.root and len(current_node.parent.data['unvisited_children']) > 0:
                    break
                elif current_node == tree.root:
                    sys.exit("The patch is NOT divisible!")
            current_node = current_node.parent.data['unvisited_children'][0]
            current_node.parent.data['unvisited_children'] = current_node.parent.data['unvisited_children'][1:]
            split_signal = True
        elif current_node.data['pivot'] is None:
            current_node, tree, split_signal = eliminateOneNode(current_node, tree, nodes_coor_lower, nodes_coor_upper)
        else:
            current_node, tree, split_signal = pivotTetGenerate(current_node, tree, nodes_coor_lower, nodes_coor_upper)
    return current_node.data['separated_tets']


def eliminateOneNode(tree_node, tree, nodes_coor_lower, nodes_coor_upper):
    assert tree_node.data['pivot'] is None
    # find all nodes with min degree
    degree_list = {}
    adj_list = tree_node.data['adj_list']
    for node, neighbors in adj_list.items():
        degree_list[node] = len(neighbors)
    deg = np.array(list(degree_list.values()))
    min_deg = min(deg)
    nodes_min_deg = [node for node, deg in degree_list.items() if deg == min_deg]
    # generate a child for every node with min degree
    for node in nodes_min_deg:
        adj_nodes_layer = [n[1] == node[1] for n in adj_list[node]]
        if False in adj_nodes_layer:
            child = Node({'adj_list': adj_list, 'boundary_tris': tree_node.data['boundary_tris'], 'pivot': node,
                          'separated_tets': tree_node.data['separated_tets'],
                          'unvisited_children': []})
            tree_node.addChild(child)
            tree_node.data['unvisited_children'].append(child)
            tree.addNode(child)
    # expand the leftmost unvisited child of tree_node
    split_signal = False
    current_node = None
    while not split_signal:
        if len(tree_node.data['unvisited_children']) != 0:
            current_node = tree_node.data['unvisited_children'][0]
            tree_node.data['unvisited_children'].remove(current_node)
            current_node, tree, split_signal = pivotTetGenerate(current_node, tree, nodes_coor_lower, nodes_coor_upper)
        else:
            return tree_node, tree, False
    return current_node, tree, True


def pivotTetGenerate(tree_node, tree, nodes_coor_lower, nodes_coor_upper):
    assert tree_node.data['pivot'] is not None
    # isolate related nodes and determine the type of pivot
    adj_list = tree_node.data['adj_list']
    boundary_tris = tree_node.data['boundary_tris']
    pivot = tree_node.data['pivot']
    # print(pivot)
    neighbors = adj_list[pivot]
    isolated_nodes = [pivot] + neighbors
    min_deg = len(neighbors)
    lower_nodes = [n for n in neighbors if n[1] == 'l']
    upper_nodes = [n for n in neighbors if n[1] == 'u']
    # check cycles around pivot node and determine its type
    if min_deg == 3:
        cycle_lengths = []
        for i, na in enumerate(neighbors):
            for nb in neighbors[i + 1:]:
                cycle = findSmallestCycle(na, nb, pivot, isolated_nodes, adj_list)
                cycle_lengths.append(len(cycle))
        num_3cycles = cycle_lengths.count(3)
        if num_3cycles == 3:
            # case 0: a free tet to take
            if len(adj_list.keys()) == 4:
                child = Node({'adj_list': [], 'boundary_tris': [], 'pivot': None,
                              'separated_tets': tree_node.data['separated_tets'] + [isolated_nodes],
                              'unvisited_children': []})
            else:
                adj_list = removeNodeInAdjacencyList(pivot, adj_list)
                try:
                    boundary_tris.remove(sortTriangleNodes(pivot, neighbors[0], neighbors[1]))
                except:
                    pass
                try:
                    boundary_tris.remove(sortTriangleNodes(pivot, neighbors[0], neighbors[2]))
                except:
                    pass
                try:
                    boundary_tris.remove(sortTriangleNodes(pivot, neighbors[1], neighbors[2]))
                except:
                    pass
                boundary_tris.append(sortTriangleNodes(neighbors[0], neighbors[1], neighbors[2]))
                child = Node({'adj_list': adj_list, 'boundary_tris': boundary_tris, 'pivot': None,
                              'separated_tets': tree_node.data['separated_tets'] + [isolated_nodes],
                              'unvisited_children': []})
                tree_node.addChild(child)
                tree.addNode(child)
            return child, tree, True
        else:
            sys.exit('Error in case 0!')
    if min_deg == 4:
        if pivot[1] == 'l':
            if len(lower_nodes) == 3:
                n3 = None
                for n in lower_nodes:
                    other_lower_nodes = lower_nodes.copy()
                    other_lower_nodes.remove(n)
                    if n in adj_list[other_lower_nodes[0]] and n in adj_list[other_lower_nodes[1]]:
                        n3 = n
                        break
                if n3 is None:
                    sys.exit('Error in cases 3 or 4!')
                else:
                    if not isSegmentInVolume(n3, upper_nodes[0], boundary_tris, nodes_coor_lower, nodes_coor_upper):
                        return tree_node, tree, False
                    connection1 = other_lower_nodes[0] in adj_list[upper_nodes[0]]
                    connection2 = other_lower_nodes[1] in adj_list[upper_nodes[0]]
                    if connection1 and connection2:
                        # case 3
                        child, tree_node, tree = removeNode([(n3, upper_nodes[0])],
                                                            [[pivot, other_lower_nodes[0], n3, upper_nodes[0]],
                                                             [pivot, other_lower_nodes[1], n3, upper_nodes[0]]],
                                                            [sortTriangleNodes(pivot, upper_nodes[0],
                                                                               other_lower_nodes[0]),
                                                             sortTriangleNodes(pivot, upper_nodes[0],
                                                                               other_lower_nodes[1])],
                                                            [sortTriangleNodes(other_lower_nodes[0], upper_nodes[0],
                                                                               n3),
                                                             sortTriangleNodes(other_lower_nodes[1], upper_nodes[0],
                                                                               n3)],
                                                            tree_node, tree, nodes_coor_lower, nodes_coor_upper)
                    elif connection1:
                        # case 4
                        child, tree_node, tree = removeNode(
                            [(n3, upper_nodes[0]), (other_lower_nodes[1], upper_nodes[0])],
                            [[pivot, other_lower_nodes[0], n3, upper_nodes[0]],
                             [pivot, other_lower_nodes[1], n3, upper_nodes[0]]],
                            [sortTriangleNodes(pivot, upper_nodes[0], other_lower_nodes[0]),
                             sortTriangleNodes(pivot, upper_nodes[0], other_lower_nodes[1])],
                            [sortTriangleNodes(other_lower_nodes[0], upper_nodes[0], n3),
                             sortTriangleNodes(other_lower_nodes[1], upper_nodes[0], n3)], tree_node, tree,
                            nodes_coor_lower, nodes_coor_upper)
                    elif connection2:
                        # case 4
                        child, tree_node, tree = removeNode(
                            [(n3, upper_nodes[0]), (other_lower_nodes[0], upper_nodes[0])],
                            [[pivot, other_lower_nodes[0], n3, upper_nodes[0]],
                             [pivot, other_lower_nodes[1], n3, upper_nodes[0]]],
                            [sortTriangleNodes(pivot, upper_nodes[0], other_lower_nodes[0]),
                             sortTriangleNodes(pivot, upper_nodes[0], other_lower_nodes[1])],
                            [sortTriangleNodes(other_lower_nodes[0], upper_nodes[0], n3),
                             sortTriangleNodes(other_lower_nodes[1], upper_nodes[0], n3)], tree_node, tree,
                            nodes_coor_lower, nodes_coor_upper)
                    else:
                        sys.exit('Error in case 3 or 4!')
                    return child, tree, True
            elif len(lower_nodes) == 2:
                # case 1
                cycle_4 = None
                for lower_n in lower_nodes:
                    for upper_n in upper_nodes:
                        if lower_n in adj_list[upper_n]:
                            other_lower_node = lower_nodes.copy()
                            other_lower_node.remove(lower_n)
                            other_upper_node = upper_nodes.copy()
                            other_upper_node.remove(upper_n)
                            if other_upper_node[0] in adj_list[upper_n] and other_lower_node[0] in adj_list[
                                lower_n] and \
                                    other_lower_node[0] in adj_list[other_upper_node[0]]:
                                cycle_4 = [lower_n, upper_n, other_upper_node[0], other_lower_node[0]]
                                break
                            else:
                                sys.exit('Error in case 1!')
                    if cycle_4 is not None:
                        break
                # check if [cycle_4[0], cycle_4[1], cycle_4[2], pivot] coplanar
                coplanar1 = is_four_points_coplanar(node2coor(cycle_4[0], nodes_coor_lower, nodes_coor_upper),
                                                    node2coor(cycle_4[1], nodes_coor_lower, nodes_coor_upper),
                                                    node2coor(cycle_4[2], nodes_coor_lower, nodes_coor_upper),
                                                    node2coor(pivot, nodes_coor_lower, nodes_coor_upper))
                # check if [cycle_4[1], cycle_4[2], cycle_4[3], pivot] coplanar
                coplanar2 = is_four_points_coplanar(node2coor(cycle_4[1], nodes_coor_lower, nodes_coor_upper),
                                                    node2coor(cycle_4[2], nodes_coor_lower, nodes_coor_upper),
                                                    node2coor(cycle_4[3], nodes_coor_lower, nodes_coor_upper),
                                                    node2coor(pivot, nodes_coor_lower, nodes_coor_upper))
                if not coplanar1:
                    add_edge1 = isSegmentInVolume(cycle_4[0], cycle_4[2], boundary_tris, nodes_coor_lower,
                                                  nodes_coor_upper)
                else:
                    add_edge1 = False
                if not coplanar2:
                    add_edge2 = isSegmentInVolume(cycle_4[1], cycle_4[3], boundary_tris, nodes_coor_lower,
                                                  nodes_coor_upper)
                else:
                    add_edge2 = False
                if add_edge1:
                    child1, tree_node, tree = removeNode([(cycle_4[0], cycle_4[2])],
                                                         [[pivot, cycle_4[0], cycle_4[1], cycle_4[2]],
                                                          [pivot, cycle_4[0], cycle_4[2], cycle_4[3]]],
                                                         [sortTriangleNodes(cycle_4[0], cycle_4[1], pivot),
                                                          sortTriangleNodes(cycle_4[1], cycle_4[2], pivot),
                                                          sortTriangleNodes(cycle_4[2], cycle_4[3], pivot)],
                                                         [sortTriangleNodes(cycle_4[0], cycle_4[1], cycle_4[2]),
                                                          sortTriangleNodes(cycle_4[0], cycle_4[2], cycle_4[3])],
                                                         tree_node, tree, nodes_coor_lower, nodes_coor_upper)
                if add_edge2:
                    child2, tree_node, tree = removeNode([(cycle_4[1], cycle_4[3])],
                                                         [[pivot, cycle_4[0], cycle_4[1], cycle_4[3]],
                                                          [pivot, cycle_4[1], cycle_4[2], cycle_4[3]]],
                                                         [sortTriangleNodes(cycle_4[0], cycle_4[1], pivot),
                                                          sortTriangleNodes(cycle_4[1], cycle_4[2], pivot),
                                                          sortTriangleNodes(cycle_4[2], cycle_4[3], pivot)],
                                                         [sortTriangleNodes(cycle_4[0], cycle_4[1], cycle_4[3]),
                                                          sortTriangleNodes(cycle_4[1], cycle_4[2], cycle_4[3])],
                                                         tree_node, tree, nodes_coor_lower, nodes_coor_upper)
                if (not add_edge1) and (not add_edge2):
                    return tree_node, tree, False
                elif add_edge1:
                    return child1, tree, True
                else:
                    return child2, tree, True
            elif len(lower_nodes) == 1:
                for upper_n in upper_nodes:
                    if upper_n not in adj_list[lower_nodes[0]]:
                        other_upper_node = upper_nodes.copy()
                        other_upper_node.remove(upper_n)
                        if other_upper_node[0] in adj_list[upper_n] and other_upper_node[1] in adj_list[upper_n]:
                            # case 2
                            coplanar1 = is_four_points_coplanar(
                                node2coor(lower_nodes[0], nodes_coor_lower, nodes_coor_upper),
                                node2coor(upper_n, nodes_coor_lower, nodes_coor_upper),
                                node2coor(other_upper_node[0], nodes_coor_lower, nodes_coor_upper),
                                node2coor(pivot, nodes_coor_lower, nodes_coor_upper))
                            coplanar2 = is_four_points_coplanar(
                                node2coor(lower_nodes[0], nodes_coor_lower, nodes_coor_upper),
                                node2coor(upper_n, nodes_coor_lower, nodes_coor_upper),
                                node2coor(other_upper_node[1], nodes_coor_lower, nodes_coor_upper),
                                node2coor(pivot, nodes_coor_lower, nodes_coor_upper))
                            cycle_coplanar = is_four_points_coplanar(
                                node2coor(lower_nodes[0], nodes_coor_lower, nodes_coor_upper),
                                node2coor(upper_n, nodes_coor_lower, nodes_coor_upper),
                                node2coor(other_upper_node[0], nodes_coor_lower, nodes_coor_upper),
                                node2coor(other_upper_node[1], nodes_coor_lower, nodes_coor_upper))
                            if coplanar1 or coplanar2:
                                return tree_node, tree, False
                            if not cycle_coplanar:
                                if not isSegmentInVolume(upper_n, lower_nodes[0], boundary_tris, nodes_coor_lower,
                                                         nodes_coor_upper):
                                    return tree_node, tree, False
                            child, tree_node, tree = removeNode([(upper_n, lower_nodes[0])],
                                                                [[pivot, upper_n, lower_nodes[0],
                                                                  other_upper_node[0]],
                                                                 [pivot, upper_n, lower_nodes[0],
                                                                  other_upper_node[1]]], [
                                                                    sortTriangleNodes(pivot, lower_nodes[0],
                                                                                      other_upper_node[0]),
                                                                    sortTriangleNodes(pivot, lower_nodes[0],
                                                                                      other_upper_node[1]),
                                                                    sortTriangleNodes(pivot, upper_n,
                                                                                      other_upper_node[0]),
                                                                    sortTriangleNodes(pivot, upper_n,
                                                                                      other_upper_node[1])], [
                                                                    sortTriangleNodes(lower_nodes[0],
                                                                                      other_upper_node[0], upper_n),
                                                                    sortTriangleNodes(lower_nodes[0],
                                                                                      other_upper_node[1], upper_n)],
                                                                tree_node, tree, nodes_coor_lower, nodes_coor_upper)
                            return child, tree, True
                        else:
                            sys.exit('Error in case 2!')
            else:
                print('Non-covered case for pivot on lower mesh!')
        else:
            if len(lower_nodes) == 1:
                n3_prime = None
                for n in upper_nodes:
                    other_upper_nodes = upper_nodes.copy()
                    other_upper_nodes.remove(n)
                    if n in adj_list[other_upper_nodes[0]] and n in adj_list[other_upper_nodes[1]]:
                        n3_prime = n
                        break
                if n3_prime is None:
                    sys.exit('Error in cases 6 or 7!')
                else:
                    if not isSegmentInVolume(n3_prime, lower_nodes[0], boundary_tris, nodes_coor_lower,
                                             nodes_coor_upper):
                        return tree_node, tree, False
                    connection1 = other_upper_nodes[0] in adj_list[lower_nodes[0]]
                    connection2 = other_upper_nodes[1] in adj_list[lower_nodes[0]]
                    if connection1 and connection2:
                        # case 6
                        child, tree_node, tree = removeNode([(n3_prime, lower_nodes[0])],
                                                            [[pivot, other_upper_nodes[0], n3_prime,
                                                              lower_nodes[0]],
                                                             [pivot, other_upper_nodes[1], n3_prime,
                                                              lower_nodes[0]]], [
                                                                sortTriangleNodes(pivot, lower_nodes[0],
                                                                                  other_upper_nodes[0]),
                                                                sortTriangleNodes(pivot, lower_nodes[0],
                                                                                  other_upper_nodes[1])], [
                                                                sortTriangleNodes(lower_nodes[0], other_upper_nodes[0],
                                                                                  n3_prime),
                                                                sortTriangleNodes(lower_nodes[0], other_upper_nodes[1],
                                                                                  n3_prime)],
                                                            tree_node, tree, nodes_coor_lower, nodes_coor_upper)
                    elif connection1:
                        # case 7
                        child, tree_node, tree = removeNode(
                            [(n3_prime, lower_nodes[0]), (other_upper_nodes[1], lower_nodes[0])],
                            [[pivot, other_upper_nodes[0], n3_prime, lower_nodes[0]],
                             [pivot, other_upper_nodes[1], n3_prime, lower_nodes[0]]],
                            [sortTriangleNodes(pivot, lower_nodes[0], other_upper_nodes[0]),
                             sortTriangleNodes(pivot, lower_nodes[0], other_upper_nodes[1])],
                            [sortTriangleNodes(lower_nodes[0], other_upper_nodes[0], n3_prime),
                             sortTriangleNodes(lower_nodes[0], other_upper_nodes[1], n3_prime)], tree_node, tree,
                            nodes_coor_lower, nodes_coor_upper)
                    elif connection2:
                        # case 7
                        child, tree_node, tree = removeNode(
                            [(n3_prime, lower_nodes[0]), (other_upper_nodes[0], lower_nodes[0])],
                            [[pivot, other_upper_nodes[0], n3_prime, lower_nodes[0]],
                             [pivot, other_upper_nodes[1], n3_prime, lower_nodes[0]]],
                            [sortTriangleNodes(pivot, lower_nodes[0], other_upper_nodes[0]),
                             sortTriangleNodes(pivot, lower_nodes[0], other_upper_nodes[1])],
                            [sortTriangleNodes(lower_nodes[0], other_upper_nodes[0], n3_prime),
                             sortTriangleNodes(lower_nodes[0], other_upper_nodes[1], n3_prime)], tree_node, tree,
                            nodes_coor_lower, nodes_coor_upper)
                    else:
                        sys.exit('Error in case 6 or 7!')
                    return child, tree, True
            elif len(lower_nodes) == 2:
                # case 5
                cycle_4 = None
                for lower_n in lower_nodes:
                    for upper_n in upper_nodes:
                        if lower_n in adj_list[upper_n]:
                            other_lower_node = lower_nodes.copy()
                            other_lower_node.remove(lower_n)
                            other_upper_node = upper_nodes.copy()
                            other_upper_node.remove(upper_n)
                            if other_upper_node[0] in adj_list[upper_n] and other_lower_node[0] in adj_list[
                                lower_n] and \
                                    other_lower_node[0] in adj_list[other_upper_node[0]]:
                                cycle_4 = [lower_n, upper_n, other_upper_node[0], other_lower_node[0]]
                                break
                            else:
                                sys.exit('Error in case 5!')
                    if cycle_4 is not None:
                        break
                # check if [cycle_4[0], cycle_4[2], cycle_4[3], pivot] coplanar
                coplanar1 = is_four_points_coplanar(node2coor(cycle_4[0], nodes_coor_lower, nodes_coor_upper),
                                                    node2coor(cycle_4[2], nodes_coor_lower, nodes_coor_upper),
                                                    node2coor(cycle_4[3], nodes_coor_lower, nodes_coor_upper),
                                                    node2coor(pivot, nodes_coor_lower, nodes_coor_upper))

                # check if [cycle_4[0], cycle_4[1], cycle_4[3], pivot] coplanar
                coplanar2 = is_four_points_coplanar(node2coor(cycle_4[0], nodes_coor_lower, nodes_coor_upper),
                                                    node2coor(cycle_4[1], nodes_coor_lower, nodes_coor_upper),
                                                    node2coor(cycle_4[3], nodes_coor_lower, nodes_coor_upper),
                                                    node2coor(pivot, nodes_coor_lower, nodes_coor_upper))
                if not coplanar1:
                    add_edge1 = isSegmentInVolume(cycle_4[0], cycle_4[2], boundary_tris, nodes_coor_lower,
                                                  nodes_coor_upper)
                else:
                    add_edge1 = False
                if not coplanar2:
                    add_edge2 = isSegmentInVolume(cycle_4[1], cycle_4[3], boundary_tris, nodes_coor_lower,
                                                  nodes_coor_upper)
                else:
                    add_edge2 = False
                if add_edge1:
                    child1, tree_node, tree = removeNode([(cycle_4[0], cycle_4[2])],
                                                         [[pivot, cycle_4[0], cycle_4[1], cycle_4[2]],
                                                          [pivot, cycle_4[0], cycle_4[2], cycle_4[3]]],
                                                         [sortTriangleNodes(cycle_4[0], cycle_4[1], pivot),
                                                          sortTriangleNodes(cycle_4[2], cycle_4[3], pivot),
                                                          sortTriangleNodes(cycle_4[3], cycle_4[0], pivot)],
                                                         [sortTriangleNodes(cycle_4[0], cycle_4[1], cycle_4[2]),
                                                          sortTriangleNodes(cycle_4[0], cycle_4[2], cycle_4[3])],
                                                         tree_node, tree, nodes_coor_lower, nodes_coor_upper)
                if add_edge2:
                    child2, tree_node, tree = removeNode([(cycle_4[1], cycle_4[3])],
                                                         [[pivot, cycle_4[0], cycle_4[1], cycle_4[3]],
                                                          [pivot, cycle_4[1], cycle_4[2], cycle_4[3]]],
                                                         [sortTriangleNodes(cycle_4[0], cycle_4[1], pivot),
                                                          sortTriangleNodes(cycle_4[2], cycle_4[3], pivot),
                                                          sortTriangleNodes(cycle_4[3], cycle_4[0], pivot)],
                                                         [sortTriangleNodes(cycle_4[0], cycle_4[1], cycle_4[3]),
                                                          sortTriangleNodes(cycle_4[1], cycle_4[2], cycle_4[3])],
                                                         tree_node, tree, nodes_coor_lower, nodes_coor_upper)
                if (not add_edge1) and (not add_edge2):
                    return tree_node, tree, False
                elif add_edge1:
                    return child1, tree, True
                else:
                    return child2, tree, True
            elif len(lower_nodes) == 3:
                for lower_n in lower_nodes:
                    if lower_n not in adj_list[upper_nodes[0]]:
                        other_lower_node = lower_nodes.copy()
                        other_lower_node.remove(lower_n)
                        if other_lower_node[0] in adj_list[upper_nodes[0]] and other_lower_node[1] in adj_list[
                            upper_nodes[0]]:
                            # case 8
                            coplanar1 = is_four_points_coplanar(
                                node2coor(upper_nodes[0], nodes_coor_lower, nodes_coor_upper),
                                node2coor(lower_n, nodes_coor_lower, nodes_coor_upper),
                                node2coor(other_lower_node[0], nodes_coor_lower, nodes_coor_upper),
                                node2coor(pivot, nodes_coor_lower, nodes_coor_upper))
                            coplanar2 = is_four_points_coplanar(
                                node2coor(upper_nodes[0], nodes_coor_lower, nodes_coor_upper),
                                node2coor(lower_n, nodes_coor_lower, nodes_coor_upper),
                                node2coor(other_lower_node[1], nodes_coor_lower, nodes_coor_upper),
                                node2coor(pivot, nodes_coor_lower, nodes_coor_upper))
                            cycle_coplanar = is_four_points_coplanar(
                                node2coor(upper_nodes[0], nodes_coor_lower, nodes_coor_upper),
                                node2coor(lower_n, nodes_coor_lower, nodes_coor_upper),
                                node2coor(other_lower_node[0], nodes_coor_lower, nodes_coor_upper),
                                node2coor(other_lower_node[1], nodes_coor_lower, nodes_coor_upper))
                            if coplanar1 or coplanar2:
                                return tree_node, tree, False
                            if not cycle_coplanar:
                                if not isSegmentInVolume(lower_n, upper_nodes[0], boundary_tris, nodes_coor_lower,
                                                         nodes_coor_upper):
                                    return tree_node, tree, False
                            child, tree_node, tree = removeNode([(lower_n, upper_nodes[0])],
                                                                [[pivot, lower_n, upper_nodes[0],
                                                                  other_lower_node[0]],
                                                                 [pivot, lower_n, upper_nodes[0],
                                                                  other_lower_node[1]]], [
                                                                    sortTriangleNodes(pivot, upper_nodes[0],
                                                                                      other_lower_node[0]),
                                                                    sortTriangleNodes(pivot, upper_nodes[0],
                                                                                      other_lower_node[1]),
                                                                    sortTriangleNodes(pivot, lower_n,
                                                                                      other_lower_node[0]),
                                                                    sortTriangleNodes(pivot, lower_n,
                                                                                      other_lower_node[1])], [
                                                                    sortTriangleNodes(upper_nodes[0],
                                                                                      other_lower_node[0], lower_n),
                                                                    sortTriangleNodes(upper_nodes[0],
                                                                                      other_lower_node[1], lower_n)],
                                                                tree_node, tree, nodes_coor_lower, nodes_coor_upper)
                            return child, tree, True
                        else:
                            sys.exit('Error in case 8!')


def boundary_edges_of_mesh(cells_idx, cells):
    '''
    Args:
        cells_idx:
        Indices of the cells composing a piece of mesh.
        cells:
        Indices of nodes on all cells.
    Returns:
        A list of unordered edges on the boundary of the piece of mesh.
    '''
    # find all edges that appear only once
    single_edges = set()
    for c in cells_idx:
        nodes = np.sort(cells[c])
        try:
            single_edges.remove((nodes[0], nodes[1]))
        except:
            single_edges.add((nodes[0], nodes[1]))
        try:
            single_edges.remove((nodes[0], nodes[2]))
        except:
            single_edges.add((nodes[0], nodes[2]))
        try:
            single_edges.remove((nodes[1], nodes[2]))
        except:
            single_edges.add((nodes[1], nodes[2]))
    return single_edges


def vtuWriteTetra(fpath: str, pts: np.array, cls: np.array, vector_fields: dict = {}):
    points = vtk.vtkPoints()
    for pt in pts:
        points.InsertNextPoint(pt[0], pt[1], pt[2])

    data_save = vtk.vtkUnstructuredGrid()
    data_save.SetPoints(points)
    for c in cls:
        cellId = vtk.vtkIdList()
        for i in c:
            cellId.InsertNextId(i)
        data_save.InsertNextCell(vtk.VTK_TETRA, cellId)

    pd = data_save.GetCellData()
    for (k, v) in vector_fields.items():
        vtk_array = numpy_support.numpy_to_vtk(v)
        vtk_array.SetName(k)
        pd.AddArray(vtk_array)

    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetInputData(data_save)
    writer.SetFileName(fpath)
    writer.Write()


if __name__ == "__main__":
    # input file processing
    parser = argparse.ArgumentParser()
    parser.add_argument('mesh', type=str, help="Path to the HDF5 file")
    parser.add_argument('-o', type=str, default="output", help="Path to the output file")
    parser.add_argument("-i", "--identical", action="store_true", help="Use identical mesh for upper and lower layers")
    parser.add_argument("-vtk", action="store_true", help="Export a .vtu file of resulting tetrahedral mesh")
    args = parser.parse_args()
    with h5py.File(args.mesh, "r") as f:
        l_nodes = f["lower_nodes"][:]
        l_cells = f["lower_cells"][:]
        next_nodes = f["next_nodes"][:]
        if args.identical:
            u_nodes = l_nodes
            u_cells = l_cells
        else:
            u_nodes = f["upper_nodes"][:]
            u_cells = f["upper_cells"][:]

    # construct helper data structures
    l_edges_set = set()
    u_edges_set = set()
    l_edge2tris_dict = {}
    u_edge2tris_dict = {}
    tri2edge_idx = set()
    tri2node_idx = set()
    for i, cl in enumerate(l_cells):
        cl = np.sort(cl)
        l_edges_set.update([(cl[0], cl[1]), (cl[0], cl[2]), (cl[1], cl[2])])
        try:
            l_edge2tris_dict[(cl[0], cl[1])].append(i)
        except:
            l_edge2tris_dict[(cl[0], cl[1])] = [i]
        try:
            l_edge2tris_dict[(cl[0], cl[2])].append(i)
        except:
            l_edge2tris_dict[(cl[0], cl[2])] = [i]
        try:
            l_edge2tris_dict[(cl[1], cl[2])].append(i)
        except:
            l_edge2tris_dict[(cl[1], cl[2])] = [i]
        tri_next_nodes = set()
        for j in range(3):
            tri_next_nodes.add(next_nodes[cl[j]])
        if len(tri_next_nodes) == 2:
            tri2edge_idx.add(i)
        elif len(tri_next_nodes) == 1:
            tri2node_idx.add(i)
    if args.identical:
        u_edges_set = l_edges_set
        u_edge2tris_dict = l_edge2tris_dict
    else:
        for i, cl in enumerate(u_cells):
            cl = np.sort(cl)
            u_edges_set.update([(cl[0], cl[1]), (cl[0], cl[2]), (cl[1], cl[2])])
            try:
                u_edge2tris_dict[(cl[0], cl[1])].append(i)
            except:
                u_edge2tris_dict[(cl[0], cl[1])] = [i]
            try:
                u_edge2tris_dict[(cl[0], cl[2])].append(i)
            except:
                u_edge2tris_dict[(cl[0], cl[2])] = [i]
            try:
                u_edge2tris_dict[(cl[1], cl[2])].append(i)
            except:
                u_edge2tris_dict[(cl[1], cl[2])] = [i]

    # identify the cutting edges
    l_cutting_edges = set()
    u_cutting_edges = set()

    for edge in l_edges_set:
        next_node1 = next_nodes[edge[0]]
        next_node2 = next_nodes[edge[1]]
        if next_node1 == next_node2:
            l_cutting_edges.add(edge)
            continue
        elif next_node1 < next_node2:
            e = (next_node1, next_node2)
        else:
            e = (next_node2, next_node1)
        if e in u_edges_set:
            l_cutting_edges.add(edge)
            u_cutting_edges.add(e)

    # divide upper and lower meshes by the cutting edges
    l_graph = nx.Graph()
    l_graph.add_nodes_from(range(l_cells.shape[0]))
    u_graph = nx.Graph()
    u_graph.add_nodes_from(range(u_cells.shape[0]))

    for edge, tris in l_edge2tris_dict.items():
        if len(tris) == 2 and edge not in l_cutting_edges:
            l_graph.add_edge(tris[0], tris[1])
    for edge, tris in u_edge2tris_dict.items():
        if len(tris) == 2 and edge not in u_cutting_edges:
            u_graph.add_edge(tris[0], tris[1])

    l_components = nx.connected_components(l_graph)
    l_components = list(l_components)
    u_components = nx.connected_components(u_graph)
    u_components = list(u_components)

    # match the upper and lower divided patches
    l_boundary_nodes = []
    u_boundary_nodes = []
    for i, comp in enumerate(l_components):
        comp_boundary = boundary_edges_of_mesh(comp, l_cells)
        comp_boundary_nodes = set()
        for edge in comp_boundary:
            if edge in l_cutting_edges:
                comp_boundary_nodes.add(next_nodes[edge[0]])
                comp_boundary_nodes.add(next_nodes[edge[1]])
        l_boundary_nodes.append(comp_boundary_nodes)
    for i, comp in enumerate(u_components):
        comp_boundary = boundary_edges_of_mesh(comp, u_cells)
        comp_boundary_nodes = set()
        for edge in comp_boundary:
            if edge in u_cutting_edges:
                comp_boundary_nodes.update(edge)
        u_boundary_nodes.append(comp_boundary_nodes)

    # form independent partitions
    independent_partitions = []
    for i, boundary_nodes in enumerate(l_boundary_nodes):
        try:
            idx = u_boundary_nodes.index(boundary_nodes)
            independent_partitions.append((i, idx))
            tri2edge_idx -= set(l_components[i])
            tri2node_idx -= set(l_components[i])
        except:
            pass

    # divide independent partitions by our node elimination algorithm
    tetrahedra = []
    for i, j in independent_partitions:
        adjacency_list, boundary_triangles = exportAdjacencyList(l_components[i], u_components[j], l_cells, u_cells,
                                                                 next_nodes)
        tetrahedra += nodeElimination(adjacency_list, boundary_triangles, l_nodes, u_nodes)
    for i in tri2edge_idx:
        cell = copy.deepcopy(l_cells[i])
        cell = np.sort(cell)
        if next_nodes[cell[0]] == next_nodes[cell[1]]:
            tetrahedra.append([(cell[0], 'l'), (cell[1], 'l'), (cell[2], 'l'), (next_nodes[cell[2]], 'u')])
            tetrahedra.append([(cell[0], 'l'), (cell[1], 'l'), (next_nodes[cell[0]], 'u'), (next_nodes[cell[2]], 'u')])
        elif next_nodes[cell[0]] == next_nodes[cell[2]]:
            tetrahedra.append([(cell[0], 'l'), (cell[1], 'l'), (cell[2], 'l'), (next_nodes[cell[0]], 'u')])
            tetrahedra.append([(cell[0], 'l'), (cell[1], 'l'), (next_nodes[cell[0]], 'u'), (next_nodes[cell[1]], 'u')])
        else:
            tetrahedra.append([(cell[0], 'l'), (cell[1], 'l'), (cell[2], 'l'), (next_nodes[cell[1]], 'u')])
    for i in tri2node_idx:
        cell = copy.deepcopy(l_cells[i])
        cell = np.sort(cell)
        tetrahedra.append([(cell[0], 'l'), (cell[1], 'l'), (cell[2], 'l'), (next_nodes[cell[0]], 'u')])

    # save file
    with open(args.o + ".pickle", 'wb') as f:
        pickle.dump(tetrahedra, f)
    if args.vtk:
        num_cells = len(tetrahedra)
        num_l_nodes = len(l_nodes)
        num_u_nodes = len(u_nodes)
        nodes = np.zeros((num_l_nodes + num_u_nodes, 3), dtype=l_nodes.dtype)
        cells = np.zeros((num_cells, 4), dtype=l_cells.dtype)
        nodes[:num_l_nodes, :2] = l_nodes
        nodes[num_l_nodes:, :2] = u_nodes
        nodes[num_l_nodes:, 2] = np.sqrt(
            max(l_nodes[:, 0].max() - l_nodes[:, 0].min(), u_nodes[:, 0].max() - u_nodes[:, 0].min()) * max(
                l_nodes[:, 1].max() - l_nodes[:, 1].min(), u_nodes[:, 1].max() - u_nodes[:, 1].min()) / (
                    len(l_cells) + len(u_cells)) * 2)
        for i, tet in enumerate(tetrahedra):
            for j, node in enumerate(tet):
                if node[1] == "l":
                    cells[i, j] = node[0]
                else:
                    cells[i, j] = node[0] + num_l_nodes
        vtuWriteTetra(args.o + ".vtu", nodes, cells)
