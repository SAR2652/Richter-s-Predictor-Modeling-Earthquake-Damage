from time import *
import math
import pandas as pd
pd.set_option('display.max_columns', 100)
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from pyspark.sql import SparkSession
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import RandomForest
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql.functions import udf, when, col, mean as _mean, stddev as _stddev
from pyspark.sql.types import IntegerType, FloatType

APP_NAME = "Richter's Predictor : Modeling Earthquake Damage"
SPARK_URL = "local[*]"
RANDOM_SEED = 42
TRAINING_DATA_RATIO = 0.75

spark = SparkSession \
    .builder \
    .appName(APP_NAME) \
    .master(SPARK_URL) \
    .getOrCreate()

# optimizations enabled by spark.sql.execution.arrow.enabled could fallback automatically to non-Arrow optimization
# implementation if an error occurs before the actual computation within Spark.
# Enable Arrow-based columnar data transfers
spark.conf.set("spark.sql.execution.arrow.enabled", "true")

pdf = pd.read_csv('data.csv')
print(pdf.head())

plt.scatter(pdf['area_percentage'], pdf['height_percentage'], c = pdf['damage_grade'])
plt.show()

plt.scatter(pdf['count_floors_pre_eq'], pdf['count_families'], c = pdf['damage_grade'])
plt.show()

sns.distplot(pdf['geo_level_3_id'], bins=21, kde=True)
plt.show()

sns.distplot(pdf['geo_level_2_id'], bins=21, kde=True)
plt.show()

# drop the 'geo_level_3_id' column due to its uniform distribution
pdf.drop(['building_id', 'geo_level_2_id', 'geo_level_3_id'], axis = 1, inplace = True)

sns.distplot(pdf['geo_level_1_id'], bins=21, kde=True)
plt.show()

# calculate number of bins using Sturge's Rule
bins = 1 + 3.322 * np.log(pdf.shape[0])

# perform binning on the column 'geo_level_1_id'
cuts = pd.cut(pdf['geo_level_1_id'], bins)
pdf.drop(['geo_level_1_id'], axis=1, inplace=True)
le = LabelEncoder()
cuts = le.fit_transform(cuts)
pdf['geo_level_1_id'] = cuts.T

# one hot encode all categorical columns
category_cols = ['geo_level_1_id', 'roof_type', 'foundation_type', 'land_surface_condition',
                 'ground_floor_type', 'other_floor_type', 'position', 'plan_configuration', 'legal_ownership_status']
 
pdf = pd.get_dummies(pdf, columns = category_cols)

# display initial 'age' column
sns.distplot(pdf['age'])
plt.show()

# display initial 'count_floors_pre_eq' column
sns.countplot(pdf['count_floors_pre_eq'])
plt.show()

# display initial 'count_families' column
sns.countplot(pdf['count_families'])
plt.show()

# convert Pandas DataFrame to Spark DataFrame
df = spark.createDataFrame(pdf)

# function to remove outliers by performing clipping using column and a threshold value
# clipping function has been converted to Pandas UDF in order to perform the operation using PySpark 
def clip_by_value(value, threshold):
    if value >= threshold:
        return threshold
    else:
        return value

def clip(threshold):
    return udf(lambda x: clip_by_value(x, threshold), IntegerType())

# print unique values of 'age' column in revers order 
unique = sorted(df.select('age').rdd.distinct().collect(), reverse = True)
uniques = [x[0] for x in unique]

print('Unique Values in Column \'age\' : {}'.format(uniques))

# apply clipping to columns 'age', 'count_floors_pre_eq' and 'count_families' values
df = df.withColumn('age', clip(200)(col("age")))\
    .withColumn('count_floors_pre_eq', clip(5)(col('count_floors_pre_eq')))\
    .withColumn('count_families', clip(4)(col('count_families')))

# display transformed plot of 'count_floors_pre_eq'
sns.countplot(df.select('count_floors_pre_eq').rdd.map(lambda x: x[0]).collect())
plt.show()

# display transformed plot of 'count_families'
sns.countplot(df.select('count_families').rdd.map(lambda x: x[0]).collect())
plt.show()

# perform log transformation on the 'area_percentage' column
df = df.withColumn('area_percentage',  udf(lambda x: math.log10(1 + x), FloatType())(col("area_percentage")))

# Standardization : Operation to transform a feature so that 
# mean = 0 and standard deviation = 1 for 
# the transformed aray of values
def standardize(x, meanVal, stdVal):
    return (x - meanVal) / stdVal

def standardize_udf(meanVal, stdVal):
    return udf(lambda x: standardize(x, meanVal, stdVal), FloatType())

# extract mean and standard deviation value for the column 'height_percentage'
df_stats_hp = df.select(_mean(col('height_percentage')).alias('mean'),
    _stddev(col('height_percentage')).alias('std')
).collect()

mean_hp = df_stats_hp[0]['mean']
std_hp = df_stats_hp[0]['std']

# extract mean and standard deviation value for the column 'age'
df_stats_age = df.select(_mean(col('age')).alias('mean'),
                     _stddev(col('age')).alias('std')
                     ).collect()

mean_age = df_stats_age[0]['mean']
std_age = df_stats_age[0]['std']

# perform simple standardization on the 'age' and 'height_percentage' column
df = df.withColumn('height_percentage', standardize_udf(mean_hp, std_hp)(col('height_percentage')))
df = df.withColumn('age', standardize_udf(mean_age, std_age)(col('age')))

# display transformed plot of 'age'
sns.distplot(df.select('age').rdd.map(lambda x: x[0]).collect())
plt.show()

# display transformed plot of 'height_percentage'
sns.distplot(df.select('height_percentage').rdd.map(lambda x: x[0]).collect(), bins=120)
plt.show()

# reduced damage grade by one as PySpark requires label encoded targets to be ordered from 0 onwards
# so damage grades will be mapped as : 1 -> 0, 2 -> 1, 3 -> 2
df = df.withColumn('damage_grade_final', udf(lambda x: x - 1, IntegerType())(col("damage_grade")))

# display transformed plot of 'damage_grade'
sns.countplot(df.select('damage_grade_final').rdd.map(lambda x: x[0]).collect())
plt.show()

# drop the old 'damage_grade' column
df = df.drop('damage_grade')

# print schema of the final dataframe
print(df.printSchema())

# print first 5 rows of final DataFrame
print(df.show(5))

# convert given spark dataframe into RDDs for effective input mapping
X_spark_rdd = df.rdd.map(lambda row: LabeledPoint(row[-1], Vectors.dense(row[0:-1])))

# Split the data into training and test sets (25% held out for testielapsed_timeng)
(X_train, X_val) = X_spark_rdd.randomSplit(
    [TRAINING_DATA_RATIO, 1 - TRAINING_DATA_RATIO])

start_time = time()

rfc = RandomForest.trainClassifier(X_train, numClasses=3, categoricalFeaturesInfo={},
                                   numTrees=15, featureSubsetStrategy="auto",
                                   impurity='gini', maxDepth=3, maxBins=92)

end_time = time()

elapsed_time = end_time - start_time
print("Time to train model: %.3f seconds" % elapsed_time)

predictions = rfc.predict(X_val.map(lambda x: x.features))
labels_and_predictions = X_val.map(lambda x: x.label).zip(predictions)

metrics = MulticlassMetrics(labels_and_predictions)
f1Score = metrics.fMeasure()

print("Evaluation Metric : F1-score")
print("F1-Score = {}".format(f1Score))
