import unittest
import findspark
import pyspark

from Transformers.FilterDepartment import FilterDepartment

"""
class PySparkTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        findspark.init()
        conf = pyspark.SparkConf().setMaster("local[*]").setAppName("testing")
        cls.sc = pyspark.SparkContext(conf=conf)
        cls.spark = pyspark.SQLContext(cls.sc)

    @classmethod
    def tearDownClass(cls):
        cls.sc.stop()
"""
"""
class test(PySparkTestCase):
    def test_with_df(self):
        df = self.spark.createDataFrame(data=[[1, 'a'], [2, 'b']],
                                        schema=['c1', 'c2'])
        self.assertEqual(df.count(), 2)

    def test_filter_department(self):
        df = pd.DataFrame(data=[["Food", "Apple"], ["Food", "Orange"], ["Game", "RDR2"]],
                          columns=["Department", "Item"])
        df = self.spark.createDataFrame(df)
        df.createOrReplaceTempView('df')

        self.assertEqual(df.count(), 3)
        filterDepartment = FilterDepartment(inputCol="Food", filterCol="Department")
        df = filterDepartment.transform(df)
        self.assertEqual(df.count(), 2)
        """