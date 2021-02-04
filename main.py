from pyspark import SparkConf
from pyspark.sql import SparkSession
from graphframes import *
import numpy as np
from pyspark.sql import functions as F
from pyspark.sql.types import *
import networkx as nx
from random_walks import RandomWalk

formatter = 'com.databricks.spark.csv'


def init_spark():
    sparkConf = SparkConf().setMaster("local[2]")
    spark1 = SparkSession\
        .builder\
        .appName("SNA")\
        .config(conf=sparkConf)\
        .getOrCreate()
    sc1 = spark1.sparkContext
    return spark1, sc1


def create_graph():
    combined = spark.read.format(formatter).options(delimiter=' ', header='false', inferSchema=True) \
        .load('facebook/facebook_combined.txt').withColumnRenamed('C0', 'src').withColumnRenamed('C1', 'dst')

    vdf = (combined.select(combined['_c0']).union(combined.select(combined['_c1']))).distinct()

    # create a dataframe with only one column
    new_vertices = vdf.select(vdf['_c0'].alias('id')).distinct()

    new_edges = combined.join(new_vertices, combined['_c0'] == new_vertices['id'], 'left')
    new_edges = new_edges.select(new_edges['_c1'], new_edges['id'].alias('src'))

    new_edges = new_edges.join(new_vertices, new_edges['_c1'] == new_vertices['id'], 'left')
    new_edges = new_edges.select(new_edges['src'], new_edges['id'].alias('dst'))

    # drop duplicate edges
    # new_edges = new_edges.dropDuplicates(['src', 'dst'])

    # print("vertex count: %d" % new_vertices.count())
    # print("edge count: %d" % new_edges.count())

    # created graph only with connections among vertices
    gf = GraphFrame(new_vertices, new_edges)
    return gf


def dictionary_function(dictionary):
    def f(x):
        return dictionary[x]
    return f


def single_random_walk(x):
    x = list(x)
    vertices = []
    neighbors_list = []
    # for each "line" in x
    # find vertex and its connections ("neighbors")
    for el in x:
        vertex = el[1][0]
        edges = el[1][1]
        vertices.append(vertex)
        neighbors = []
        for edge in edges:
            if edge[1] in vertices:
                neighbors.append(edge[1])
        neighbors_list.append(neighbors)

    edges = x[0][1][1]
    g = nx.Graph(edges)
    random_walk = RandomWalk(g, walk_length=5, num_walks=1, p=10, q=10, workers=4, quiet=True)
    walk_nodes = np.unique(np.array(random_walk.walks).reshape(-1))

    # select a random node to start with // or just start with the first node
    # but make sure that this node is in this partition!
    start_vertex = np.random.choice(vertices)
    # print("min == ",min(vertices),", max == ",max(vertices),", start node == ",start_vertex)
    visited = []  # keep a list with the visited nodes, through the whole process of random walk
    len_walk = 10  # sample 100 nodes in each walk
    for c in range(1, len_walk + 1):
        position = vertices.index(start_vertex)
        vertex_neighbors = neighbors_list[position]
        if len(vertex_neighbors) == 0:
            continue
        probability = []  # probability of visiting a neighbor is uniform
        probability = probability + [1. / len(vertex_neighbors)] * len(vertex_neighbors)
        start_vertex = np.random.choice(vertex_neighbors, p=probability)

        if start_vertex in visited:
            continue
        else:
            visited.append(start_vertex)
    print(walk_nodes.shape, len(vertices), len(visited))
    return visited


if __name__ == '__main__':
    spark, sc = init_spark()
    sc.setLogLevel("ERROR")

    graph = create_graph()  # facebook graph, only vertices and edges, without metadata for each vertex

    communities = graph.labelPropagation(maxIter=10)
    print(f"There are {communities.select('label').distinct().count()} communities in sample graph.")

    normalizedCommunities = list(communities.select('label').distinct().toLocalIterator())
    d = dict()
    for i in range(len(normalizedCommunities)):
        d.update({normalizedCommunities[i].__getattr__('label'): i})
    numPartitions = len(d)

    dict_f = dictionary_function(d)
    udf_dict = F.udf(dict_f, IntegerType())

    df = communities.withColumn('label', udf_dict('label'))

    V = graph.vertices.rdd.map(lambda x: x['id']).sortBy(lambda x: x)
    E1 = graph.edges.rdd.map(lambda x: (x['src'], x['dst']))
    E2 = graph.edges.rdd.map(lambda x: (x['dst'], x['src']))
    grouped_E1 = E1.groupBy(lambda x: x[0]).map(lambda x: (x[0], list(x[1])))
    grouped_E2 = E2.groupBy(lambda x: x[0]).map(lambda x: (x[0], list(x[1])))
    grouped_E = grouped_E1.union(grouped_E2).groupByKey().map(lambda x: (x[0], (list(x[1]))[0])).repartition(2)

    p_v = df.rdd.map(lambda x: (x[1], x[0])).repartition(2)
    p_v_e = p_v.map(lambda x: (x[1], x[0])).join(grouped_E).map(lambda x: (x[1][0], (x[0], x[1][1])))
    partitioned_vertices = p_v_e.partitionBy(numPartitions)
    sampled = partitioned_vertices.mapPartitions(lambda x: single_random_walk(x))
    print(sampled.count())
