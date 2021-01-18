from pyspark import SparkConf
from pyspark.sql import SparkSession
from graphframes import *
from pyspark.sql import functions as F
from pyspark.sql.types import *
from random import sample

formatter = 'com.databricks.spark.csv'


def init_spark():
    sparkConf = SparkConf().setMaster("local[2]")
    spark = SparkSession\
        .builder\
        .appName("SNA")\
        .config(conf=sparkConf)\
        .getOrCreate()
    sc = spark.sparkContext
    return spark, sc


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
    graph = GraphFrame(new_vertices, new_edges)
    return graph


def load_dummy_graph():
    # define vertices
    vertices = spark.createDataFrame([
        ("a", "Alice", 34),
        ("b", "Bob", 36),
        ("c", "Charlie", 30),
        ("d", "David", 29),
        ("e", "Esther", 32),
        ("f", "Fanny", 36),
        ("g", "Gabby", 60)], ["id", "name", "age"])

    # define edges
    edges = spark.createDataFrame([
        ("a", "b", "follow"),
        ("b", "c", "follow"),
        ("c", "b", "follow"),
        ("c", "e", "follow"),
        ("f", "c", "follow"),
        ("e", "f", "follow"),
        ("e", "d", "follow"),
        ("d", "a", "follow"),
        ("a", "e", "follow")
    ], ["src", "dst", "relationship"])
    # create graph
    graph = GraphFrame(vertices, edges)
    return graph


if __name__ == '__main__':
    spark, sc = init_spark()
    sc.setLogLevel("ERROR")

    dummy = load_dummy_graph()
    """
    Some basic functions:
    g = dummy #our graph
    g.vertices.show() #print vertices
    g.edges.show() #print edges
    
    # calculate inDegrees for vertices
    vertexInDegrees = g.inDegrees
    vertexInDegrees.show() #you can also use .show(k), with k:int, the number of results that you want to take
    
    # Find the youngest user's age in the graph.
    g.vertices.groupBy().min("age").show()
    
    # Count the number of "follows" in the graph.
    numFollows = g.edges.filter("relationship = 'follow'").count()
    print(numFollows)
    
    # Count the number of triangles and show in how many triangles each vertex belongs.
    results = g.triangleCount()
    results.select("id", "count").show()
    """

    graph = create_graph()  # facebook graph, only vertices and edges, without metadata for each vertex

    # graph.inDegrees.show(5)

    communities = graph.labelPropagation(maxIter=5)
    # communities.persist().show()

    normalizedCommunities = list(communities.select('label').distinct().toLocalIterator())
    d = dict()
    for i in range(len(normalizedCommunities)):
        d.update({normalizedCommunities[i].__getattr__('label'): i})

    def dictionary_function(d):
        def f(x):
            return d[x]
        return f
    dict_f = dictionary_function(d)
    udf_dict = F.udf(dict_f, IntegerType())

    df = communities.withColumn('label', udf_dict('label'))

    df = df.rdd.map(lambda x: (x[1], x[0])).partitionBy(len(d))

    def foo(x):
        #print(type(x))
        x1 = list(x)
        #Set = set()
        #for s in x:
        #    Set.add(s[0])
        #print(Set)
        return sample(x1, min(len(x1), 10))

    df2 = df.mapPartitions(foo)
    print(df2.count())

    print(f"There are {communities.select('label').distinct().count()} communities in sample graph.")
