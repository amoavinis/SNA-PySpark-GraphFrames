from pyspark import SparkConf
from pyspark.sql import SparkSession
from graphframes import *
import numpy as np
from pyspark.sql import functions as F
from pyspark.sql.types import *
import networkx as nx
import matplotlib.pyplot as plt

formatter = 'com.databricks.spark.csv'


def init_spark():
    sparkConf = SparkConf().setMaster("local[2]")
    spark1 = SparkSession \
        .builder \
        .appName("SNA") \
        .config(conf=sparkConf) \
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

    # created graph only with connections among vertices
    gf = GraphFrame(new_vertices, new_edges)
    return gf


def dictionary_function(dictionary):
    def f(x):
        return dictionary[x]

    return f


def get_len(n_nodes, cc, alpha):
    return n_nodes / (1 + alpha*cc)


def single_random_walk(x, alpha):
    x = list(x)
    vertices = []
    edges_list = []
    neighbors_list = []
    # for each "line" in x
    # find vertex and its connections ("neighbors")
    for el in x:
        vertex = el[1][0]
        edges = el[1][1]
        edges_list.extend(edges)
        vertices.append(vertex)
        neighbors = []
        for edge in edges:
            neighbors.append(edge[1])
        neighbors_list.append(neighbors)

    for index, vertex in enumerate(vertices):
        temp_list = []
        for dst in neighbors_list[index]:
            if dst in vertices:
                temp_list.append(dst)
        neighbors_list[index] = temp_list

    # compute average clustering coefficient
    g = nx.Graph(edges_list)
    cc = nx.average_clustering(g)

    # select a random node to start with // or just start with the first node
    # but make sure that this node is in this partition!
    start_vertex = np.random.choice(vertices)

    visited = list()  # keep a list with the visited nodes, through the whole process of random walk
    visited.append(start_vertex)
    len_walk = int(get_len(len(vertices), cc, alpha)) + 1  # sample n nodes in each walk

    for c in range(1, len_walk):
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

    return visited

def plot_degree_dist(G):
    degrees = [G.degree(n) for n in G.nodes()]
    plt.hist(degrees)
    plt.show()


if __name__ == '__main__':
    spark, sc = init_spark()
    sc.setLogLevel("ERROR")

    graph = create_graph()  # facebook graph, only vertices and edges, without metadata for each vertex

    a = 2
    max_iter = 5

    nx_graph = nx.Graph(graph.edges.rdd.collect())

    plot_original = False
    if plot_original:
        nx.draw(nx_graph)
        plt.savefig('original_graph.eps', format='eps')

    calculate_original = False
    if calculate_original:
        # node degree distribution
        hist = nx.degree_histogram(nx_graph)
        deg = [i for i in range(0,len(hist))]
        pk = [(i/len(hist)) for i in hist]
        plt.bar(deg,pk, width=0.8)
        plt.savefig('original_histogram.eps', format='eps')

        # average clustering coefficient -- global
        print("Average clustering coefficient:", nx.average_clustering(nx_graph))

        # average degree
        degrees = nx.degree(nx_graph)
        avg_deg = sum(dict(degrees).values()) / len(degrees)
        print("Average degree:", avg_deg)

        # average betweenness centrality
        bc = nx.betweenness_centrality(nx_graph)
        res = sum(bc.values()) / len(bc)
        print("Average betweenness centrality", res)

        print("Diameter:", nx.diameter(nx_graph))

        # average closeness centrality
        cl = nx.closeness_centrality(nx_graph)
        print("Average closeness centrality:", sum(cl.values())/len(cl))

        # transitivity
        t = nx.transitivity(nx_graph)
        print("Transitivity:", t)

    communities = graph.labelPropagation(maxIter=max_iter)
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
    sampled = partitioned_vertices.mapPartitions(lambda x: single_random_walk(x, a))

    # keep only distinct nodes
    v = sc.parallelize(np.unique(np.array(sampled.collect())))
    edges_rdd = graph.edges.rdd.map(lambda x: (x['src'], x['dst']))

    # find all possible pairs between vertices
    combinations = v.cartesian(v).filter(lambda x: x[0] != x[1]) \
        .map(lambda x: (x, 0))

    e = combinations.join(edges_rdd.map(lambda x: (x, 0))).map(lambda x: x[0]).collect()

    new_graph = nx.Graph(e)

    # compute same properties for sampled graph, in order to compare it with the original
    make_histogram = True
    if make_histogram:
        hist = nx.degree_histogram(new_graph)
        deg = [i for i in range(0, len(hist))]
        pk = [(i / len(hist)) for i in hist]
        plt.bar(deg, pk, width=0.8)
        plt.savefig('sampled_histogram_max'+str(max_iter)+'_a'+str(a)+'.eps', format='eps')

    calculate_metrics = False
    if calculate_metrics:
        # average clustering coefficient -- global
        print("Average clustering coefficient:", nx.average_clustering(new_graph))

        degrees = nx.degree(new_graph)
        avg_deg = sum(dict(degrees).values()) / len(degrees)
        print("Average degree:", avg_deg)

        bc = nx.betweenness_centrality(new_graph)
        res = sum(bc.values()) / len(bc)
        print("Average betweenness centrality", res)

        t = nx.transitivity(new_graph)
        print("Transitivity:", t)

        cl = nx.closeness_centrality(new_graph)
        print("Average closeness centrality:", sum(cl.values()) / len(cl))

    make_graph = False
    if make_graph:
        nx.draw(new_graph)
        plt.savefig('sampled_graph_max'+str(max_iter)+'_a'+str(a)+'.eps', format='eps')
