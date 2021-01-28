from pyspark import SparkConf
from pyspark.sql import SparkSession
from graphframes import *
from pyspark.sql import functions as F
from pyspark.sql.types import *

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


if __name__ == '__main__':
    spark, sc = init_spark()
    sc.setLogLevel("ERROR")

    graph = create_graph()  # facebook graph, only vertices and edges, without metadata for each vertex

    communities = graph.labelPropagation(maxIter=5)
    print(f"There are {communities.select('label').distinct().count()} communities in sample graph.")

    normalizedCommunities = list(communities.select('label').distinct().toLocalIterator())
    d = dict()
    for i in range(len(normalizedCommunities)):
        d.update({normalizedCommunities[i].__getattr__('label'): i})
    numPartitions = len(d)

    def dictionary_function(dictionary):
        def f(x):
            return dictionary[x]
        return f
    dict_f = dictionary_function(d)
    udf_dict = F.udf(dict_f, IntegerType())

    df = communities.withColumn('label', udf_dict('label'))

    V = graph.vertices.rdd.map(lambda x: x['id']).sortBy(lambda x: x)
    E1 = graph.edges.rdd.map(lambda x: (x['src'], x['dst']))
    E2 = graph.edges.rdd.map(lambda x: (x['dst'], x['src']))
    grouped_E1 = E1.groupBy(lambda x: x[0]).map(lambda x: (x[0], list(x[1])))
    grouped_E2 = E2.groupBy(lambda x: x[0]).map(lambda x: (x[0], list(x[1])))
    grouped_E = grouped_E1.union(grouped_E2).groupByKey().map(lambda x: (x[0], (list(x[1]))[0])).repartition(2)


    def random_walk(x):
        x = list(x)
        vertices = x[0][0]
        edges = x[0][1]
        #print("Doing random walk here...")

        return [0]


    p_v = df.rdd.map(lambda x: (x[1], x[0])).repartition(2)
    p_v_e = p_v.map(lambda x: (x[1], x[0])).join(grouped_E).map(lambda x: (x[1][0], (x[0], x[1][1])))
    partitioned_vertices = p_v_e.partitionBy(numPartitions)
    sampled = partitioned_vertices.mapPartitions(lambda x: random_walk(x))
    print(sampled.count())

