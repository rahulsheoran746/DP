# from pyspark.sql import SparkSession
# from pyspark.sql.functions import explode,col,udf,to_json,collect_list
# from pyspark.sql.types import StringType,StructField,StructType,ArrayType
# import json
# import boto3
# from pyspark.accumulators import AccumulatorParam


# class StringAccumulator(AccumulatorParam):
#     def zero(self, initialValue=""):
#         return initialValue
 
#     def addInPlace(self, v1, v2):
#         return v1 + "\n" + v2 if v1 and v2 else v1 or v2



# spark = SparkSession.builder\
#     .appName('SparkTest')\
#     .config('spark.hadoop.fs.s3a.access.key','')\
#     .config('spark.hadoop.fs.s3a.secret.key','')\
#     .getOrCreate()

# file_path = 's3a://cdp-raw-us-b24/test_rahul/silver_raw_data/*.txt'
# write_path = 's3a://cdp-raw-us-b24/test_rahul/silver_final_data/testSpark'  
# spark.sparkContext.setLogLevel("FATAL")
# print('Spark session started')

# # Create an accumulator to collect malformed JSONs
# malformed_json_accumulator = spark.sparkContext.accumulator("", StringAccumulator())

# def convert_complex_columns_to_json(df):
#     for field in df.schema.fields:
#         if isinstance(field.dataType, (ArrayType, StructType)):
#             df = df.withColumn(field.name, to_json(col(field.name)))
#     return df

# # Function to write malformed JSONs to S3
# def write_malformed_jsons_to_s3(json_set):
#     s3 = boto3.client('s3',
#                       aws_access_key_id='',
#                       aws_secret_access_key='')
#     file_key = f"{write_path}/malformed_jsons.txt"
#     bucket_name = 'cdp-raw-us-b24'
#     json_str = "\n".join(json_set)
#     try:
#         # Write the JSON string to S3
#         s3.put_object(Bucket=bucket_name, Key=file_key, Body=json_str)
#         print(f"Malformed JSON written to S3: s3://{bucket_name}/{file_key}")
#     except Exception as e:
#         print(f"Error writing malformed JSON to S3: {e}")

# # Function to segregate JSONs and extract eventType
# def segregate_jsons_and_extract_eventType(text):
#     jsons = []
#     stack = []
#     start_index = 0
#     for i, char in enumerate(text):
#         if char == '{':
#             if not stack:
#                 start_index = i
#             stack.append('{')
#         elif char == '}' and stack:
#             stack.pop()
#             if not stack:
#                 segment = text[start_index:i+1]
#                 try:
#                     json_obj = json.loads(segment)
#                     event_type = json_obj.get('eventType', json_obj.get('eventtype', 'nullType'))
#                     jsons.append((event_type, json.dumps(json_obj)))
#                 except json.JSONDecodeError as e:
#                     print('malformed json is:', segment)
#                     malformed_json_accumulator.add(segment)
#     return jsons
 
# segregate_jsons_and_extract_eventType_udf = udf(segregate_jsons_and_extract_eventType,ArrayType(\
#     StructType([\
#         StructField('eventType',StringType()),\
#         StructField('json_string',StringType())\
#                 ])))

# df = spark.read.text(file_path,wholetext = True)
# df = df.withColumn('value',explode(segregate_jsons_and_extract_eventType_udf(col('value'))))
# df = df.withColumn('eventType',col('value')['eventType']).\
#         withColumn('json_string', col('value')['json_string']).drop('value')
# df.show(truncate = False)
# # df.show(truncate = False)
# event_types = df.select("eventType").distinct().collect()
# for event_type_row in event_types:
#         event_type = event_type_row["eventType"]
#         event_json_df = df.filter(col("eventType") == event_type).select('eventType',"json_string")
#         event_df = spark.read.json(event_json_df.rdd.map(lambda x: x['json_string']))
#         event_df = convert_complex_columns_to_json(event_df) 
#         # Write to CSV
#         csv_output_path = f"{write_path}/{event_type}"
#         # event_df.write.option("header", "true").option("quoteAll","true").option("escape","\"").mode("overwrite").csv(csv_output_path)
#         event_df.coalesce(1).write.csv(csv_output_path,mode = 'overwrite',header = 'true',quoteAll = 'true',escape = "\"") 
#         event_df.show(truncate = False)
#         print(f"CSV for eventType '{event_type}' written to: {csv_output_path}")

# print(malformed_json_accumulator.value)
# # Write all accumulated malformed JSONs to S3
# if malformed_json_accumulator.value.strip():
#     print('Some malformed json found in this data ')
#     malformed_jsons = set(malformed_json_accumulator.value.strip().split("\n"))
#     write_malformed_jsons_to_s3(malformed_jsons)


# # code with multithreading and accumulator is defined globally

# import json
# from pyspark import SparkConf
# from pyspark.sql import SparkSession
# from pyspark import SparkConf, AccumulatorParam
# from pyspark.sql.functions import udf, explode, col, to_json
# from pyspark.sql.types import StringType, ArrayType, StructType, StructField, Row
# from datetime import timedelta, datetime
# import concurrent.futures
# import boto3

# # Custom Accumulator for collecting strings (malformed JSONs)
# class StringAccumulator(AccumulatorParam):
#     def zero(self, initialValue=""):
#         return initialValue
 
#     def addInPlace(self, v1, v2):
#         return v1 + "\n" + v2 if v1 and v2 else v1 or v2

# # Configure Spark settings
# conf = SparkConf().setAppName("JSON Segregation and Processing")
# # Initialize Spark Session with the configurations
# # spark = SparkSession.builder.config(conf=conf).getOrCreate()
# spark = SparkSession.builder \
#                 .appName("ReadCSVAndWriteWithSpark") \
#                 .config("spark.hadoop.fs.s3a.access.key", "") \
#                 .config("spark.hadoop.fs.s3a.secret.key", "")\
#                 .getOrCreate()
# sc = spark.sparkContext  # Get the underlying SparkContext
# sc.setLogLevel("FATAL")

# # Define the input and output base paths
# # input_base_path = "/mnt/dam-prod-events"
# # output_base_path = "/mnt/cdp-raw-us-b24/dam-events"
# input_base_path = "s3a://events-prod"
# output_base_path = "s3a://cdp-raw-us-b24/test_rahul/silver_final_data/testSpark"

# # Global variable for collecting malformed JSONs
# malformed_json_accumulator = sc.accumulator("", StringAccumulator())

# # Function to convert complex columns to JSON
# def convert_complex_columns_to_json(df):
#     for field in df.schema.fields:
#         if isinstance(field.dataType, (ArrayType, StructType)):
#             df = df.withColumn(field.name, to_json(col(field.name)))
#     return df

# # Function to write malformed JSONs to S3
# def write_malformed_jsons_to_s3(failure_records_path,json_set):
#     s3 = boto3.client('s3',
#                       aws_access_key_id='',
#                       aws_secret_access_key='')
#     file_key = f"{failure_records_path}/malformed_jsons.txt"
#     bucket_name = 'cdp-raw-us-b24'
#     json_str = "\n".join(json_set)
#     try:
#         # Write the JSON string to S3
#         s3.put_object(Bucket=bucket_name, Key=file_key, Body=json_str)
#         print(f"Malformed JSON written to S3: s3://{bucket_name}/{file_key}")
#     except Exception as e:
#         print(f"Error writing malformed JSON to S3: {e}")

# # Function to segregate JSONs and extract eventType
# def segregate_jsons_and_extract_eventType(text):
#     jsons = []
#     stack = []
#     start_index = 0
#     for i, char in enumerate(text):
#         if char == '{':
#             if not stack:
#                 start_index = i
#             stack.append('{')
#         elif char == '}' and stack:
#             stack.pop()
#             if not stack:
#                 segment = text[start_index:i+1]
#                 try:
#                     json_obj = json.loads(segment)
#                     event_type = json_obj.get('eventType', json_obj.get('eventtype', 'nullType'))
#                     jsons.append((event_type, json.dumps(json_obj)))
#                 except json.JSONDecodeError as e:
#                     malformed_json_accumulator.add(segment)
#     return jsons

# # Register UDF for segregating JSONs and extracting eventType
# segregate_jsons_and_extract_eventType_udf = udf(segregate_jsons_and_extract_eventType, ArrayType(StructType([
#     StructField("eventType", StringType(), True),
#     StructField("json_string", StringType(), True)
# ])))

# # Function to process data for each hour
# def process_data_for_hour(hour):
#     input_json_path = f"{input_base_path}/{yesterday_date_for_source}/{hour:02d}"
#     output_csv_path = f"{output_base_path}/{yesterday_date_for_source}/{hour:02d}"
    
#     # Read JSON files as text
#     df = spark.read.text(input_json_path, wholetext=True)
#     # df.show(truncate = False)
    
#     # Segregate JSONs and extract eventType
#     json_rdd = df.select(explode(segregate_jsons_and_extract_eventType_udf("value")).alias("data")).rdd
#     json_df = spark.createDataFrame(json_rdd.map(lambda x: Row(eventType=x['data']['eventType'], json_string=x['data']['json_string'])))
#     # json_df.show(truncate = False)
    
#     # Get unique event types
#     event_types = json_df.select("eventType").distinct().collect()
    
#     # Process each event type
#     for event_type_row in event_types:
#         event_type = event_type_row["eventType"]
#         if event_type == 'bidrequest':
#             event_json_df = json_df.filter(col("eventType") == event_type).select("json_string")
#             event_df = spark.read.json(event_json_df.rdd.map(lambda x: x['json_string']))
#             event_df = convert_complex_columns_to_json(event_df)
            
#             # Write to CSV
#             csv_output_path = f"{output_csv_path}/{event_type}"
#             # event_df.show(truncate = False)
#             event_df.write.option("header", "true").option("quoteAll","true").option('escape','\"').mode("overwrite").csv(csv_output_path)
#             print(f"CSV for eventType '{event_type}' written to: {csv_output_path}")

# # Function to write accumulated malformed JSONs to a file
# def write_malformed_jsons_to_file(date):
#     # failure_records_path = f"/mnt/cdp-raw-us-b24/dam-events/{date}/failure_records"  # Path for storing malformed JSONs
#     failure_records_path = f"events-prod/{date}/failure_records"  # Path for storing malformed JSONs
#     if malformed_json_accumulator.value.strip():
#         print('Some malformed json found in this date ',date)
#         # os.makedirs(failure_records_path, exist_ok=True)
#         unique_malformed_jsons = set(malformed_json_accumulator.value.strip().split("\n"))
#         # with open(os.path.join(failure_records_path, "malformed_jsons.txt"), "w") as f:
#         #     f.write("\n".join(unique_malformed_jsons))
#         write_malformed_jsons_to_s3(failure_records_path,unique_malformed_jsons)
    
#     # Reset accumulator for the next day
#     malformed_json_accumulator.add("appending data for date",date)

# # Define start and end dates
# start_date = '2024/01/01'
# end_date = '2024/01/01'

# # Iterate over each day
# current_date = datetime.strptime(start_date, '%Y/%m/%d')
# end_date = datetime.strptime(end_date, '%Y/%m/%d')

# while current_date <= end_date:
#     # Global variable for collecting malformed JSONs
#     # malformed_json_accumulator = sc.accumulator("", StringAccumulator())
    
#     yesterday_date_for_source = current_date.strftime('%Y/%m/%d')
#     with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
#         # Submit tasks for each hour
#         futures = [executor.submit(process_data_for_hour, hour) for hour in range(4,6)]
        
#         # Wait for all tasks to complete
#         for future in concurrent.futures.as_completed(futures):
#             # Get any exceptions raised by the task
#             if future.exception() is not None:
#                 print(f"Error occurred: {future.exception()}")
    
#     # Write accumulated malformed JSONs for the day to a file
#     write_malformed_jsons_to_file(yesterday_date_for_source)
    
#     # Move to the next day
#     current_date += timedelta(days=1)

# # Stop Spark Session
# spark.stop()

 
# import json
# from pyspark import SparkConf
# from pyspark.sql import SparkSession
# from pyspark import SparkConf, AccumulatorParam
# from pyspark.sql.functions import udf, explode, col, to_json
# from pyspark.sql.types import StringType, ArrayType, StructType, StructField, Row
# from datetime import timedelta, datetime
# import concurrent.futures
# import boto3
 
# # Custom Accumulator for collecting strings (malformed JSONs)
# class StringAccumulator(AccumulatorParam):
#     def zero(self, initialValue=""):
#         return initialValue
 
#     def addInPlace(self, v1, v2):
#         return v1 + "\n" + v2 if v1 and v2 else v1 or v2
 
# # Configure Spark settings
# conf = SparkConf().setAppName("JSON Segregation and Processing")
# # Initialize Spark Session with the configurations
# # spark = SparkSession.builder.config(conf=conf).getOrCreate()
# spark = SparkSession.builder \
#                 .appName("ReadCSVAndWriteWithSpark") \
#                 .config("spark.hadoop.fs.s3a.access.key", "") \
#                 .config("spark.hadoop.fs.s3a.secret.key", "")\
#                 .config("spark.hadoop.fs.s3a.endpoint", "s3-us-east-2.amazonaws.com") \
#                 .getOrCreate()
# sc = spark.sparkContext  # Get the underlying SparkContext
# sc.setLogLevel("FATAL")
 
# # Define the input and output base paths
# # input_base_path = "/mnt/dam-prod-events"
# # output_base_path = "/mnt/cdp-raw-us-b24/dam-events"
# input_base_path = "s3a://events-prod"
# output_base_path = "s3a://cdp-raw-us-b24/events-prod2"
 
# # Global variable for collecting malformed JSONs
# malformed_json_accumulator = sc.accumulator("", StringAccumulator())
 
# # Function to convert complex columns to JSON
# def convert_complex_columns_to_json(df):
#     for field in df.schema.fields:
#         if isinstance(field.dataType, (ArrayType, StructType)):
#             df = df.withColumn(field.name, to_json(col(field.name)))
#     return df
 
# # Function to write malformed JSONs to S3
# def write_malformed_jsons_to_s3(failure_records_path,json_set):
#     s3 = boto3.client('s3',
#                       aws_access_key_id='',
#                       aws_secret_access_key='')
#     file_key = f"{failure_records_path}/malformed_jsons.txt"
#     bucket_name = 'cdp-raw-us-b24'
#     json_str = "\n".join(json_set)
#     try:
#         # Write the JSON string to S3
#         s3.put_object(Bucket=bucket_name, Key=file_key, Body=json_str)
#         print(f"Malformed JSON written to S3: s3://{bucket_name}/{file_key}")
#     except Exception as e:
#         print(f"Error writing malformed JSON to S3: {e}")
 
# # Function to segregate JSONs and extract eventType
# def segregate_jsons_and_extract_eventType(text):
#     jsons = []
#     stack = []
#     start_index = 0
#     for i, char in enumerate(text):
#         if char == '{':
#             if not stack:
#                 start_index = i
#             stack.append('{')
#         elif char == '}' and stack:
#             stack.pop()
#             if not stack:
#                 segment = text[start_index:i+1]
#                 try:
#                     json_obj = json.loads(segment)
#                     event_type = json_obj.get('eventType', json_obj.get('eventtype', 'nullType'))
#                     if not event_type: 
#                         event_type = 'nullType'
#                     jsons.append((event_type, json.dumps(json_obj)))
#                 except json.JSONDecodeError as e:
#                     malformed_json_accumulator.add(segment)
#     return jsons
 
# # Register UDF for segregating JSONs and extracting eventType
# segregate_jsons_and_extract_eventType_udf = udf(segregate_jsons_and_extract_eventType, ArrayType(StructType([
#     StructField("eventType", StringType(), True),
#     StructField("json_string", StringType(), True)
# ])))
 
# # Function to process data for each hour
# def process_data_for_hour(hour):
#     input_json_path = f"{input_base_path}/{yesterday_date_for_source}/{hour:02d}"
#     output_csv_path = f"{output_base_path}/{yesterday_date_for_source}/{hour:02d}"
 
   
#     # Read JSON files as text
#     df = spark.read.text(input_json_path, wholetext=True)
   
#     # Segregate JSONs and extract eventType
#     json_rdd = df.select(explode(segregate_jsons_and_extract_eventType_udf("value")).alias("data")).rdd
#     json_df = spark.createDataFrame(json_rdd.map(lambda x: Row(eventType=x['data']['eventType'], json_string=x['data']['json_string'])))
   
#     # Get unique event types
#     event_types = json_df.select("eventType").distinct().collect()
#     print('event type is ',event_types)
   
#     # Process each event type
#     for event_type_row in event_types:
#         event_type = event_type_row["eventType"]
#         event_json_df = json_df.filter(col("eventType") == event_type).select("json_string")
#         event_df = spark.read.json(event_json_df.rdd.map(lambda x: x['json_string']))
#         event_df = convert_complex_columns_to_json(event_df)
    
#         # Write to CSV
#         csv_output_path = f"{output_csv_path}/{event_type}"
#         print(event_type)
#         print(f'Count of no of rows for eventtype {event_type} is :',event_df.count())
#         # event_df.write.option("header", "true").option("quoteAll","true").option('escape', '\"').mode("overwrite").csv(csv_output_path)
#         print(f"CSV for eventType '{event_type}' written to: {csv_output_path}")
 
# # Function to write accumulated malformed JSONs to a file
# def write_malformed_jsons_to_file(date):
#     # failure_records_path = f"/mnt/cdp-raw-us-b24/dam-events/{date}/failure_records"  # Path for storing malformed JSONs
#     failure_records_path = f"events-prod2/{date}/failure_records"  # Path for storing malformed JSONs
#     if malformed_json_accumulator.value.strip():
#         print('Some malformed json found in this date ',date)
#         # os.makedirs(failure_records_path, exist_ok=True)
#         unique_malformed_jsons = set(malformed_json_accumulator.value.strip().split("\n"))
#         # with open(os.path.join(failure_records_path, "malformed_jsons.txt"), "w") as f:
#         #     f.write("\n".join(unique_malformed_jsons))
#         write_malformed_jsons_to_s3(failure_records_path,unique_malformed_jsons)
   
#     # Reset accumulator for the next day
#     message = f"Appending data for date {date}"
#     malformed_json_accumulator.add(message)
 
# # Define start and end dates
# start_date = '2024/01/01'
# end_date = '2024/01/01'
 
# # Iterate over each day
# current_date = datetime.strptime(start_date, '%Y/%m/%d')
# end_date = datetime.strptime(end_date, '%Y/%m/%d')
 
# while current_date <= end_date:
#     # Global variable for collecting malformed JSONs
#     # malformed_json_accumulator = sc.accumulator("", StringAccumulator())
   
#     yesterday_date_for_source = current_date.strftime('%Y/%m/%d')
#     with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
#         # Submit tasks for each hour
#         futures = [executor.submit(process_data_for_hour, hour) for hour in range(5, 6)]
#         # Wait for all tasks to complete
#         for future in concurrent.futures.as_completed(futures):
#             # Get any exceptions raised by the task
#             if future.exception() is not None:
#                 print(f"Error occurred: {future.exception()}")
#     # for hour in range(4, 6):
#     #     process_data_for_hour(hour)
   
#     # Write accumulated malformed JSONs for the day to a file
#     write_malformed_jsons_to_file(yesterday_date_for_source)
   
#     # Move to the next day
#     current_date += timedelta(days=1)
 
# # Stop Spark Session
# spark.stop()



# import json
# from pyspark import SparkConf
# from pyspark.sql import SparkSession
# from pyspark import SparkConf, AccumulatorParam
# from pyspark.sql.functions import udf, explode, col, to_json
# from pyspark.sql.types import StringType, ArrayType, StructType, StructField, Row
# from datetime import timedelta, datetime
# import concurrent.futures
# import boto3
 
# # Custom Accumulator for collecting strings (malformed JSONs)
# class StringAccumulator(AccumulatorParam):
#     def zero(self, initialValue=""):
#         return initialValue
 
#     def addInPlace(self, v1, v2):
#         return v1 + "\n" + v2 if v1 and v2 else v1 or v2
 
# # Configure Spark settings
# conf = SparkConf().setAppName("JSON Segregation and Processing")
# # Initialize Spark Session with the configurations
# # spark = SparkSession.builder.config(conf=conf).getOrCreate()
# spark = SparkSession.builder \
#                 .appName("ReadCSVAndWriteWithSpark") \
#                 .config("spark.hadoop.fs.s3a.access.key", "") \
#                 .config("spark.hadoop.fs.s3a.secret.key", "")\
#                 .config("spark.hadoop.fs.s3a.endpoint", "s3-us-east-2.amazonaws.com") \
#                 .getOrCreate()
# sc = spark.sparkContext  # Get the underlying SparkContext
# sc.setLogLevel("FATAL")
 
# # Define the input and output base paths
# # input_base_path = "/mnt/dam-prod-events"
# # output_base_path = "/mnt/cdp-raw-us-b24/dam-events"
# input_base_path = "s3a://cdp-raw-us-b24/test_rahul/silver_raw_data/"
# output_base_path = "s3a://cdp-raw-us-b24/events-prod"
 
# # Global variable for collecting malformed JSONs
# malformed_json_accumulator = sc.accumulator("", StringAccumulator())
 
# # Function to convert complex columns to JSON
# def convert_complex_columns_to_json(df):
#     for field in df.schema.fields:
#         if isinstance(field.dataType, (ArrayType, StructType)):
#             df = df.withColumn(field.name, to_json(col(field.name)))
#     return df
 
# # Function to write malformed JSONs to S3
# def write_malformed_jsons_to_s3(failure_records_path,json_set):
#     s3 = boto3.client('s3',
#                       aws_access_key_id='',
#                       aws_secret_access_key='')
#     file_key = f"{failure_records_path}/malformed_jsons.txt"
#     bucket_name = 'cdp-raw-us-b24'
#     json_str = "\n".join(json_set)
#     try:
#         # Write the JSON string to S3
#         s3.put_object(Bucket=bucket_name, Key=file_key, Body=json_str)
#         print(f"Malformed JSON written to S3: s3://{bucket_name}/{file_key}")
#     except Exception as e:
#         print(f"Error writing malformed JSON to S3: {e}")
 
# # Function to segregate JSONs and extract eventType
# def segregate_jsons_and_extract_eventType(text):
#     jsons = []
#     stack = []
#     start_index = 0
#     for i, char in enumerate(text):
#         if char == '{':
#             if not stack:
#                 start_index = i
#             stack.append('{')
#         elif char == '}' and stack:
#             stack.pop()
#             if not stack:
#                 segment = text[start_index:i+1]
#                 try:
#                     json_obj = json.loads(segment)
#                     event_type = json_obj.get('eventType', json_obj.get('eventtype', 'nullType'))
#                     # if not event_type: 
#                     #     event_type = 'nullType'
#                     jsons.append((event_type, json.dumps(json_obj)))
#                 except json.JSONDecodeError as e:
#                     malformed_json_accumulator.add(segment)
#     return jsons
 
# # Register UDF for segregating JSONs and extracting eventType
# segregate_jsons_and_extract_eventType_udf = udf(segregate_jsons_and_extract_eventType, ArrayType(StructType([
#     StructField("eventType", StringType(), True),
#     StructField("json_string", StringType(), True)
# ])))
 
# # Function to process data for each hour
# def process_data_for_hour(hour):
#     input_json_path = f"{input_base_path}/{yesterday_date_for_source}/{hour:02d}"
#     output_csv_path = f"{output_base_path}/{yesterday_date_for_source}/{hour:02d}"
 
   
#     # Read JSON files as text
#     df = spark.read.text(input_json_path, wholetext=True)
   
#     # Segregate JSONs and extract eventType
#     json_rdd = df.select(explode(segregate_jsons_and_extract_eventType_udf("value")).alias("data")).rdd
#     json_df = spark.createDataFrame(json_rdd.map(lambda x: Row(eventType=x['data']['eventType'], json_string=x['data']['json_string'])))
   
#     # Get unique event types
#     event_types = json_df.select("eventType").distinct().collect()
#     print('event type is ',event_types)
   
#     # Process each event type
#     for event_type_row in event_types:
#         event_type = event_type_row["eventType"]
#         if event_type != '':
#             # if event_type == 'bidrequest':
#             event_json_df = json_df.filter(col("eventType") == event_type).select("json_string")
#             event_df = spark.read.json(event_json_df.rdd.map(lambda x: x['json_string']))
#             event_df = convert_complex_columns_to_json(event_df)
        
#             # Write to CSV
#             csv_output_path = f"{output_csv_path}/{event_type}"
#             # print(event_type)
#             # print(event_df.count())
#             event_df.write.option("header", "true").option("quoteAll","true").option('escape', '\"').mode("overwrite").csv(csv_output_path)

        
#             print(f"CSV for eventType '{event_type}' written to: {csv_output_path}")
 
# # Function to write accumulated malformed JSONs to a file
# def write_malformed_jsons_to_file(date):
#     # failure_records_path = f"/mnt/cdp-raw-us-b24/dam-events/{date}/failure_records"  # Path for storing malformed JSONs
#     failure_records_path = f"events-prod9/{date}/failure_records"  # Path for storing malformed JSONs
#     if malformed_json_accumulator.value.strip():
#         print('Some malformed json found in this date ',date)
#         # os.makedirs(failure_records_path, exist_ok=True)
#         unique_malformed_jsons = set(malformed_json_accumulator.value.strip().split("\n"))
#         # with open(os.path.join(failure_records_path, "malformed_jsons.txt"), "w") as f:
#         #     f.write("\n".join(unique_malformed_jsons))
#         write_malformed_jsons_to_s3(failure_records_path,unique_malformed_jsons)
   
#     # Reset accumulator for the next day
#     message = f"Appending data for date {date}"
#     malformed_json_accumulator.add(message)
 
# # Define start and end dates
# start_date = '2024/01/01'
# end_date = '2024/01/02'
 
# # Iterate over each day
# current_date = datetime.strptime(start_date, '%Y/%m/%d')
# end_date = datetime.strptime(end_date, '%Y/%m/%d')
 
# while current_date <= end_date:
#     # Global variable for collecting malformed JSONs
#     # malformed_json_accumulator = sc.accumulator("", StringAccumulator())
   
#     yesterday_date_for_source = current_date.strftime('%Y/%m/%d')
#     # with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
#     #     # Submit tasks for each hour
#     #     futures = [executor.submit(process_data_for_hour, hour) for hour in range(4, 5)]
#     #     # Wait for all tasks to complete
#     #     for future in concurrent.futures.as_completed(futures):
#     #         # Get any exceptions raised by the task
#     #         if future.exception() is not None:
#     #             print(f"Error occurred: {future.exception()}")
#     for hour in range(4, 5):
#         process_data_for_hour(hour)
   
#     # Write accumulated malformed JSONs for the day to a file
#     write_malformed_jsons_to_file(yesterday_date_for_source)
   
#     # Move to the next day
#     current_date += timedelta(days=1)
 
# # Stop Spark Session
# spark.stop()


# from pyspark.sql import SparkSession

# spark = SparkSession.builder\
#     .appName('SparkTest')\
#     .config('spark.hadoop.fs.s3a.access.key','')\
#     .config('spark.hadoop.fs.s3a.secret.key','')\
#     .getOrCreate()
# spark.sparkContext.setLogLevel("FATAL")
# print('Spark session started')
# output_path = 's3a://cdp-silver-us-b24/test/spark_test/'

# data = [(1,'Rahul'),(2,'Mohit'),(3,'ANkit')]

# df = spark.createDataFrame(data,['id','name'])
# df.show()
# df.coalesce(1).write.csv(output_path,header = 'true',mode = 'overwrite')
# print('csv file written successfully')




# import logging
# from pyspark.sql import SparkSession
# from pyspark.sql.functions import *
# from datetime import datetime, timedelta
# import time


# # Create a Spark session
# spark = SparkSession.builder \
#     .appName("DHC Data To GOLD") \
#     .config("spark.hadoop.fs.s3a.access.key",'' )\
#     .config("spark.hadoop.fs.s3a.secret.key","" )\
#     .getOrCreate()
# spark.sparkContext.setLogLevel("FATAL")
# logging.info('Spark session started successfully.')
# bucket_name = 'cdp-silver-us-b24'
# base_folder_name = 'dhc-silver-output'
# lst = ['Hospital_Clinical_Trials_Details', 'Hospital_Executives', 'Hospital_Inpatient_Diagnoses_ICD_by_Attending_Physician_Most_Recent_Full_Year','Hospital_Inpatient_Diagnoses_ICD_by_Attending_Physician_Most_Recent_YTD', 'Hospital_Outpatient_Diagnoses_ICD_by_Attending_Physician_Most_Recent_Full_Year','Hospital_Outpatient_Diagnoses_ICD_by_Attending_Physician_Most_Recent_YTD','Hospital_Outpatient_Procedures_HCPCS_CPT_by_Attending_Physician_Most_Recent_Full_Year','Hospital_Outpatient_Procedures_HCPCS_CPT_by_Attending_Physician_Most_Recent_YTD', 'Hospital_Outpatient_Procedures_HCPCS_CPT_by_Operating_Physician_Most_Recent_Full_Year','Hospital_Outpatient_Procedures_HCPCS_CPT_by_Operating_Physician_Most_Recent_YTD','Hospital_Overview', 'Hospital_Physician_Affiliations_Current','Hospital_Top_Referring_Physicians','Physician_Emails','Physicians_Board_Certifications','Physicians_Claims_Based_Specialties','Physicians_Overview']
# def path_dhc_data(bucket_name, base_folder_name, lst):
#     paths = []
#     for name in lst:
#         path = f's3a://{bucket_name}/{base_folder_name}/{name}/year=2024/month=05/day=21/hour=__HIVE_DEFAULT_PARTITION__/*.snappy.parquet'
#         paths.append(path)
#     return paths

# paths = path_dhc_data(bucket_name, base_folder_name, lst)
# for path in paths:
#     print(path)

# # Stop the Spark session
# logging.info("Stopping the Spark session.")
# spark.stop()
# logging.info("Spark session stopped successfully.")

# from pyspark.sql import SparkSession
# from pyspark.sql.functions import *

# spark = spark = SparkSession.builder \
#     .appName("verify data for GOLD") \
#     .config("spark.hadoop.fs.s3a.access.key",'' )\
#     .config("spark.hadoop.fs.s3a.secret.key","" )\
#     .getOrCreate()
# spark.sparkContext.setLogLevel('FATAL')

# input_path = f"s3a://cdp-silver-us-b24/gold/active_npi_per_day/year=2024/month=01/day=*/*.snappy.parquet"

# df = spark.read.parquet(input_path, header = 'true')
# print(df.count())
# df.show(truncate = False)
# df = df.select('npi', 'cookie_id').distinct()
# wrong_df = df.filter((col('cookie_id') == 'undefined'))
# wrong_df.show(truncate = False)
# print(df.count())

# input_path = f"s3a://cdp-silver-us-b24/gold/final_table_with_brand_affinity1/year=2024/month=01/day=*/*.snappy.parquet"

# df = spark.read.parquet(input_path, header = 'true')
# print(df.count())
# df.show(truncate = False)
# df = df.select('npi').distinct()
# print(df.count())



# from pyspark.sql import SparkSession
# from pyspark.sql.functions import col, length, to_date, explode, array, \
#     split, regexp_replace, lit, when, substring_index, from_unixtime
# from pyspark.sql.types import StructType, StructField, StringType, TimestampType
# from datetime import datetime, timedelta

# # Create Spark Session
# def create_spark_session():
#     spark = SparkSession.builder \
#         .appName("Data Processing Application") \
#         .config("spark.hadoop.fs.s3a.access.key", "") \
#         .config("spark.hadoop.fs.s3a.secret.key", "") \
#         .getOrCreate()
#     return spark
# spark = create_spark_session()
# spark.sparkContext.setLogLevel('FATAL')
# spark.sql("set spark.sql.legacy.timeParserPolicy=LEGACY")


# # Define paths and columns
# bidrequest_base_path = "s3a://cdp-silver-us-b24/events-prod/bidrequest1/year=2024/month=01/day={}/"
# bidrequest_columns = ["docereeCookiePlatformUid", "eventUnixTimeStampMs", 'user_platformUid', 'platformUid']

# openrtb_bidrequest_base_path = "s3a://cdp-silver-us-b24/events-prod/openrtb_bidrequest/year=2024/month=01/day={}/"
# openrtb_bidrequest_columns = ["docereeCookiePlatformUid", "eventUnixTimeStampMs", 'user_platformUid', 'platformUid']


# for day in range(1, 2):
#     day_str = str(day).zfill(2)
#     bidrequest_path = bidrequest_base_path.format(day_str)
#     openrtb_bidrequest_base_path = openrtb_bidrequest_base_path.format(day_str)

#      # Read data
#     try:
#         filtered_ad_request = spark.read.format("parquet") \
#             .option("header", "true") \
#             .option('inferSchema', 'true') \
#             .load(bidrequest_path) \
#             .select(bidrequest_columns) \
#             .filter(
#                 (
#                     (col("docereeCookiePlatformUid").isNotNull()) & 
#                     (length(col("docereeCookiePlatformUid")) > 0) & 
#                     (col("docereeCookiePlatformUid") != 'NULL') & 
#                     (col("docereeCookiePlatformUid") != '')
#                 ) | (
#                     (col("user_platformUid").isNotNull()) & 
#                     (length(col("user_platformUid")) > 0) & 
#                     (col("user_platformUid") != 'NULL') & 
#                     (col("user_platformUid") != '')
#                 ) | (
#                     (col("platformUid").isNotNull()) & 
#                     (length(col("platformUid")) > 0) & 
#                     (col("platformUid") != 'NULL') & 
#                     (col("platformUid") != '')
#                 )
#             ) \
#             .filter(
#                 to_date(from_unixtime(col("eventUnixTimeStampMs") / 1000), 'yyyy-MM-dd') == '2024-01-01'
#             ) \
#             .withColumn('created_at', to_date(from_unixtime(col("eventUnixTimeStampMs") / 1000), 'yyyy-MM-dd'))
            
#         print('adrequest are:', filtered_ad_request.count())

        
#     except Exception as e:
#         print(f"Data for bidrequest not found for day {day_str}: {e}")
#         filtered_ad_request = None

    # try:
    #     opentrb_filtered_ad_request = spark.read.format("parquet") \
    #         .option("header", "true") \
    #         .option('inferSchmea','true')\
    #         .load(openrtb_bidrequest_base_path) \
    #         .select(openrtb_bidrequest_columns) \
    #         .filter((col("user_platformUid").isNotNull()) & (length(col("user_platformUid")) > 0) & (col("user_platformUid") != 'NULL') |
    #                 (col("platformUid").isNotNull()) & (length(col("platformUid")) > 0) & (col("platformUid") != 'NULL')) \
    #         .withColumn("created_at", to_date((col("eventUnixTimeStampMs") / 1000).cast("timestamp"))).dropDuplicates()
    # except Exception as e:
    #     print(f"Data for openrtb_bidrequest not found for day {day_str}: {e}")
    #     opentrb_filtered_ad_request = None

    # Process data if available
    # if filtered_ad_request is not None:
    #     df1 = filtered_ad_request.select(col("docereeCookiePlatformUid").alias("cookie_id"),'created_at').filter((col("docereeCookiePlatformUid").isNotNull()) & (col("user_platformUid") != 'undefined')).filter(col('created_at') == '2024-01-01').distinct()
    #     df2 = filtered_ad_request.select(col("user_platformUid").alias("cookie_id"),'created_at').filter((col("user_platformUid").isNotNull()) & (col("user_platformUid") != 'undefined')).filter(col('created_at') == '2024-01-01').distinct()
    #     df3 = filtered_ad_request.select(col("platformUid").alias("cookie_id"),'created_at').filter((col("platformUid").isNotNull()) & (col("platformUid") != 'undefined')).filter(col('created_at') == '2024-01-01').distinct()
    #     filtered_ad_request = df1.unionByName(df2).unionByName(df3).distinct()

    #     print('filtered_ad_request count:', filtered_ad_request.count())


# from pyspark.sql import SparkSession
# from pyspark.sql.functions import col, date_format
# from datetime import datetime, timedelta
# import time
# # Initialize Spark session

# spark = SparkSession.builder \
#     .appName("SnowflakeToS3") \
#     .config("spark.hadoop.fs.s3a.access.key", "") \
#     .config("spark.hadoop.fs.s3a.secret.key", "") \
#     .config("spark.sql.files.maxPartitionBytes", "134217728") \
#     .config("spark.sql.files.minPartitionNum", "1") \
#     .config("spark.sql.shuffle.partitions", "40") \
#     .getOrCreate()
# spark.sparkContext.setLogLevel('FATAL')
# print('Spark session created successfully.')

# # Snowflake connection options
# snowflake_options = {
#     "sfURL": "https://ip85260.us-east-2.aws.snowflakecomputing.com",
#     "sfWarehouse": "COMPUTE_WH",
#     "sfDatabase": "DOCEREE_ANALYTICS",
#     "sfSchema": "PLATFORM",
#     "sfRole": "PROD_READ_ONLY",
#     "sfUser": "DS_PROD",
#     "sfPassword": "N6yrGyxbkH"
# }

# # S3 bucket name
# output_base_path = "s3a://cdp-raw-us-b24/test_rahul/ad_request/csv4"

# # Function to fetch data from Snowflake for a specific day
# def fetch_data_for_day(date_str):
#     query = f"SELECT * FROM ad_request WHERE DATE(created_at) = '{date_str}'"
#     df = spark.read.format("snowflake") \
#         .option("header", "true")\
#         .option("quote", "\"")\
#         .option("escape", "\"")\
#         .option("multiline", "true")\
#         .options(**snowflake_options) \
#         .option("query", query) \
#         .load()
#     print(f'df after reading data for {date_str}:', df.count())
#     # df.show(truncate = False)
#     return df


# # Function to write data to S3
# def write_data_to_s3(df, date_str):
#     date_folder = datetime.strptime(date_str, '%Y-%m-%d').strftime('%Y/%m/%d')
#     # print('df before writing :', df.count())
#     # df.show(truncate = False)
#     output_path = f"{output_base_path}/{date_folder}/"
#     df.write.option('header', 'true').option('quoteAll', 'true').option("escape", "\"").mode("overwrite").parquet(output_path)
#     print(f'Data for {date_str} written to {output_path}')

# start_time = time.time()
# # Main script
# start_date = datetime(2024, 1, 1)  # Start date
# end_date = datetime(2024, 1, 2)  # End date

# current_date = start_date
# while current_date <= end_date:
#     date_str = current_date.strftime("%Y-%m-%d")
#     # print(f"Processing data for date: {date_str}")
    
#     # Fetch data
#     df = fetch_data_for_day(date_str)
    
#     # Write data to S3
#     write_data_to_s3(df, date_str)
    
#     # Move to the next day
#     current_date += timedelta(days=1)


# end_time = time.time()
# time_taken = end_time- start_time
# print(f'total time taken is : {time_taken} seconds')
# # Stop the Spark session
# spark.stop()



# from pyspark.sql import SparkSession
# from pyspark.sql.functions import col, date_format, length
# from datetime import datetime, timedelta
# from concurrent.futures import ThreadPoolExecutor, as_completed
# import time
# import logging

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger()
# # Initialize Spark session
# spark = SparkSession.builder \
#     .appName("SnowflakeToS3") \
#     .config("spark.hadoop.fs.s3a.access.key", "") \
#     .config("spark.hadoop.fs.s3a.secret.key", "") \
#     .config("spark.sql.files.maxPartitionBytes", str(128 * 1024 * 1024)) \
#     .config("spark.sql.shuffle.partitions", "200") \
#     .getOrCreate()

# spark.sparkContext.setLogLevel('FATAL')
# logger.info('Spark session created successfully.')

# # Snowflake connection options
# snowflake_options = {
#     "sfURL": "https://ip85260.us-east-2.aws.snowflakecomputing.com",
#     "sfWarehouse": "COMPUTE_WH",
#     "sfDatabase": "DOCEREE_ANALYTICS",
#     "sfSchema": "PLATFORM",
#     "sfRole": "PROD_READ_ONLY",
#     "sfUser": "DS_PROD",
#     "sfPassword": "N6yrGyxbkH"
# }


# output_base_path = "s3a://cdp-raw-us-b24/snowflake/adrequest/test"


# def fetch_data_for_day(date_str):
#     try:
#         query = f"SELECT * FROM ad_request WHERE DATE(created_at) = '{date_str}'"
#         df = spark.read.format("snowflake") \
#             .option("header", "true") \
#             .option("quote", "\"") \
#             .option("escape", "\"") \
#             .option("multiline", "true") \
#             .options(**snowflake_options) \
#             .option("query", query) \
#             .load()
#         count = df.count()
#         logger.info(f'df after reading data for {date_str}: {count} records found')
#         df.printSchema()
#         return df
#     except Exception as e:
#         logger.error(f'Error fetching data for {date_str}: {e}')
#         return None


# def write_data_to_s3(df, date_str):
#     try:
#         # target_partition_size_mb = 128
#         # total_size_bytes = df.rdd.map(lambda row: len(str(row))).sum()
#         # print(f'total data size in bytes for {date_str} is :',total_size_bytes)
#         # num_partitions = max(1, total_size_bytes // (target_partition_size_mb * 1024 * 1024))

#         # df = df.repartition(num_partitions)
#         date_folder = datetime.strptime(date_str, '%Y-%m-%d').strftime('%Y/%m/%d')
#         output_path = f"{output_base_path}/{date_folder}/"
#         # df.write.option('header', 'true') \
#         #     .option('quoteAll', 'true') \
#         #     .option("escape", "\"") \
#         #     .mode("overwrite") \
#         #     .csv(output_path)
#         logger.info(f'Data for {date_str} written to {output_path}')
#     except Exception as e:
#         logger.error(f'Error writing data for {date_str}: {e}')
     
# def perform_data_analysis(df):
#     logger.info('current Schema for this df is:')
#     # print('count is: ',df.count())
#     # df.printSchema()
#     # df.show(truncate = False)
#     columns_to_select = ['HCP_ID', 'PLATFORMUID', 'USERPLATFORMUID', 'DOCEREECOOKIEPLATFORMUID', 'USERNPI', 'USEREMAIL', 'USERZIPCODE']

#     df = df.filter((length(col('HCP_ID')) != 10) & (col("HCP_ID").isNotNull()) & (col('HCP_ID') != '')).select('USERPLATFORMUID','HCP_ID')
#     print('count for column hcp_id where it is not number: ', df.count())
#     df.show(truncate = False)
#     df = df.filter(col('USERPLATFORMUID').isNull())
#     df.show(truncate = False)
#     # df1 = df.filter((col('USERPLATFORMUID').isNotNull()) & (col('USERPLATFORMUID') != '') & (col('USERPLATFORMUID') != 'NULL') & (col('USERPLATFORMUID') != ' ')  & (col('USERPLATFORMUID') != 'undefined')).select(col('USERPLATFORMUID').alias("cookie_id")).distinct()
#     # # df1 = df1.filter(length(col('USERPLATFORMUID')) != 36).select('USERPLATFORMUID')
#     # print('count of userplatformuid:', df1.count())
#     # # df1.show(truncate = False)
#     # df2 = df.filter((col('PLATFORMUID').isNotNull()) & (col('PLATFORMUID') != '') & (col('PLATFORMUID') != 'NULL') & (col('PLATFORMUID') != ' ')  & (col('PLATFORMUID') != 'undefined')).select(col('PLATFORMUID').alias("cookie_id")).distinct()
#     # # df2 = df2.filter(length(col('PLATFORMUID')) != 36).select('PLATFORMUID')
#     # print('count of PLATFORMUID :', df2.count())
#     # # df2.show(truncate = False)
#     # df3 = df.filter((col('DOCEREECOOKIEPLATFORMUID').isNotNull()) & (col('DOCEREECOOKIEPLATFORMUID') != '') & (col('DOCEREECOOKIEPLATFORMUID') != 'NULL') & (col('DOCEREECOOKIEPLATFORMUID') != ' ')  & (col('DOCEREECOOKIEPLATFORMUID') != 'undefined')).select(col('DOCEREECOOKIEPLATFORMUID').alias("cookie_id")).distinct()
#     # # df3 = df3.filter(length(col('DOCEREECOOKIEPLATFORMUID')) != 36).select('DOCEREECOOKIEPLATFORMUID')
#     # print('count of DOCEREECOOKIEPLATFORMUID :', df3.count())
#     # # df1.show(truncate = False)
#     # # df = df.filter((col('HCP_ID').isNotNull()) & (col('HCP_ID') != '')).select(columns_to_select)
#     # # df.show(truncate = False)
#     # filtered_ad_request = df1.unionByName(df2).unionByName(df3).distinct()
#     # print('all cookie id are:', filtered_ad_request.count())

#     return

# # Function to process data for a single day
# def process_day(date_str):
#     df = fetch_data_for_day(date_str)
#     if df:
#         # perform_data_analysis(df)
#         write_data_to_s3(df, date_str)
#     else:
#         logger.warning(f'No data to process for {date_str}')

# # Main script
# start_date = datetime(2024, 1, 1)  # Start date
# end_date = datetime(2024, 1, 1)  # End date

# #without multithreading
# # current_date = start_date
# # while current_date <= end_date:
# #     date_str = current_date.strftime("%Y-%m-%d")
# #     process_day(date_str)
# #     # Move to the next day
# #     current_date += timedelta(days=1)

# # with multithreading
# dates = []
# current_date = start_date
# while current_date <= end_date:
#     dates.append(current_date.strftime("%Y-%m-%d"))
#     current_date += timedelta(days=1)

# start_time = time.time()
# # Use ThreadPoolExecutor to process dates concurrently
# with ThreadPoolExecutor(max_workers=10) as executor:
#     futures = [executor.submit(process_day, date) for date in dates]
#     for future in as_completed(futures):
#         try:
#             future.result()
#         except Exception as e:
#             print(f'Error processing data: {e}')

# end_time = time.time()
# total_time = end_time - start_time
# print(f"Total time taken: {total_time} seconds")
# # Stop the Spark session
# spark.stop()


# from pyspark.sql import SparkSession

# # Initialize Spark session
# spark = SparkSession.builder \
#     .appName("S3 to Snowflake") \
#     .config("spark.hadoop.fs.s3a.access.key", "") \
#     .config("spark.hadoop.fs.s3a.secret.key", "") \
#     .getOrCreate()
# spark.sparkContext.setLogLevel('FATAL')

# # Define S3 path



# s3_path = "s3a://cdp-silver-us-b24/gold/test/*.parquet"

# # Read data from S3
# df = spark.read.parquet(s3_path)

# print("READ SUCCESSFUL >>>>>>>>>>")
# # Show the data
# df.show(truncate=False)

# # Snowflake connection options
# sf_options = {
#     "sfURL": "ip85260.us-east-2.aws.snowflakecomputing.com",
#     "sfUser": "rudder",
#     "sfPassword": "Changeit@123",
#     "sfDatabase": "RUDDERDB",
#     "sfSchema": "WRITE_TEST",
#     "sfWarehouse": "RUDDERSTACK",
#     "sfRole": "RUDDER_RW"
# }

# # Write data to Snowflake
# df.write \
#     .format("net.snowflake.spark.snowflake") \
#     .options(**sf_options) \
#     .option("dbtable", "NPI_AM_DATA") \
#     .mode("overwrite") \
#     .save()


# dmd_validated_hcp latest data script using SFTP
# import paramiko

# ssh_client = paramiko.SSHClient()

# host = 'ftpserver.devops.doceree.com'
# username = 'dmdfeedfile'
# password = 'S9B0S07rz3oq'
# port = 21
# try:
#     ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
#     ssh_client.connect(hostname = host, username = username, password = password, port = port)
#     print('Connection Established Successfully')

#     ftp = ssh_client.open_sftp()
#     files = ftp.listdir()
#     print('List all directory:', files)

#     ftp.close()
#     ssh_client.close()
# except Exception as e :
#     print("Inside catch block")
#     print(e)


# from ftplib import FTP_TLS
 
# # FTPS server details
# ftp_server = "ftpserver.devops.doceree.com"
# ftp_user = "dmdfeedfile"
# ftp_password = "S9B0S07rz3oq"
 
# ftps = FTP_TLS(ftp_server)
 
# # Secure the connection
# ftps.auth()  # Secure the FTP connection using TLS
 
 
# ftps.login(user=ftp_user, passwd=ftp_password)
 
 
# ftps.prot_p()
 
# print("Secure connection established...")
# print("Current Directory:", ftps.pwd())

# #List the files in the current directory
# files = ftps.nlst()
# print("Files in the current directory:")
# for file in files:
#     print(file)

# ftps.quit()
 
 
# if __name__ == "__main__":
#     print("TEST")



# import ftplib
# import ssl

# def connect_to_ftp(host, username, password):
#     try:

#         # context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT) 
#         # # context.minimum_version = ssl.TLSVersion.TLSv1_3
#         # context.check_hostname = True  # Enable hostname checking
#         # context.verify_mode = ssl.CERT_REQUIRED 

#         # Connect to the FTP server
#         ftp = ftplib.FTP_TLS()
#         ftp.ssl_version = ssl.TLSVersion.TLSv1_3
      
        
#         # ftp.ssl_context = context
#         ftp.debugging = 2
#         ftp.connect(host = host, port= 21)

#         ftp.auth()
        
#         # Login to the FTP server
#         ftp.login(user=username, passwd=password)

#         ftp.prot_p()
#         print("Secure connection established...")
        
#         # Enable passive mode
#         ftp.set_pasv(True)

#         print("Current Directory:", ftp.pwd())
#         # ftp.sendport(host = host, port= 21)
        
#         # ftp.retrlines('LIST') 

#         # ftp.dir()
        
#        # List the files in the current directory
#         try:
#             files = ftp.mlsd()
#             print("Files in the current directory:")
#             for file in files:
#                 print(file)
#         except ftplib.error_perm as e:
#             print(f"FTP permission error: {e}")
#         except Exception as e:
#             print(f"Error listing files: {e}")
        
#         # Close the connection
#         ftp.quit()
#         print("Connection closed.")
#     except ftplib.all_errors as e:
#         print(f"FTP error: {e}")


# host = 'ftpserver.devops.doceree.com'
# username = 'rahul'
# password = 'rahul'

# connect_to_ftp(host, username, password)

# import ftplib
# import ssl

# def connect_to_ftp(host, username, password):
#     try:
#         ftp = ftplib.FTP_TLS()
#         ftp.ssl_version = ssl.TLSVersion.TLSv1_3

#         # Enable debugging to see what's happening behind the scenes
#         ftp.debugging = 2
        
#         # Connect to the FTP server
#         ftp.connect(host=host, port=21)
#         ftp.auth()
        
#         # Login to the FTP server
#         ftp.login(user=username, passwd=password)

#         # Protect data connections
#         ftp.prot_p()
#         print("Secure connection established...")
        
#         # Enable passive mode
#         ftp.set_pasv(True)

#         print("Current Directory:", ftp.pwd())

#         # List the files in the current directory
#         try:
#             # Explicitly reuse the control connection for data transfer
#             files = ftp.mlsd()
#             print("Files in the current directory:")
#             for file in files:
#                 print(file)
#         except ftplib.error_perm as e:
#             print(f"FTP permission error: {e}")
#         except Exception as e:
#             print(f"Error listing files: {e}")
        
#         # Close the connection
#         ftp.quit()
#         print("Connection closed.")
#     except ftplib.all_errors as e:
#         print(f"FTP error: {e}")


# host = 'ftpserver.devops.doceree.com'
# username = 'rahul'
# password = 'rahul'

# connect_to_ftp(host, username, password)



# import ftplib
# import ssl

# def connect_to_ftps(host, username, password):
#     try:
#         # Create an SSL context forcing TLS version
#         context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
#         context.check_hostname = False
#         context.verify_mode = ssl.CERT_NONE
        
#         # Connect to the FTP server using FTP_TLS with the specified SSL context
#         ftps = ftplib.FTP_TLS(context=context)
#         ftps.debugging = 2
        
#         # Connect to the host
#         ftps.connect(host, 21)
        
        
#         # Enable secure data connection
#         ftps.auth()
        
#         # Login to the FTP server
#         ftps.login(user=username, passwd=password)
        
#         # Switch to secure mode for the data connection
#         ftps.prot_p()
        
#         # Enable passive mode
#         ftps.set_pasv(True)
        
#         # List the files in the current directory
#         files = ftps.nlst()
#         print("Files in the current directory:")
#         for file in files:
#             print(file)
        
#         # Close the connection
#         ftps.quit()
#         print("Connection closed.")
#     except ftplib.all_errors as e:
#         print(f"FTP error: {e}")

# host = 'ftpserver.devops.doceree.com'
# username = 'vishvendra'
# password = 'Changeit@123'

# connect_to_ftps(host, username, password)


# import ftplib
# import os
# from datetime import datetime

# FTP_HOST = "ftp.ed.ac.uk"
# FTP_USER = "anonymous"
# FTP_PASS = ""

# def get_size_format(n, suffix="B"):
#     # converts bytes to scaled format (e.g KB, MB, etc.)
#     for unit in ["", "K", "M", "G", "T", "P"]:
#         if n < 1024:
#             return f"{n:.2f}{unit}{suffix}"
#         n /= 1024

# def get_datetime_format(date_time):
#     # convert to datetime object
#     date_time = datetime.strptime(date_time, "%Y%m%d%H%M%S")
#     # convert to human readable date time string
#     return date_time.strftime("%Y/%m/%d %H:%M:%S")

# ftp = ftplib.FTP(FTP_HOST, FTP_USER, FTP_PASS)

# ftp.encoding = "utf-8"

# print(ftp.getwelcome())


# ftp.cwd("pub/maps")

# print("*"*50, "LIST", "*"*50)
# ftp.dir()

# ftp.quit()



# from pyspark.sql import SparkSession, Row
# from datetime import datetime, timedelta
# import logging 
# from pyspark.sql.functions import *

# logger = logging.getLogger()
# logging.basicConfig(level=logging.INFO)
# # Initialize Spark session
# spark = SparkSession.builder \
#     .appName("Daily Refresh Count") \
#     .config("spark.hadoop.fs.s3a.access.key", "") \
#     .config("spark.hadoop.fs.s3a.secret.key", "") \
#     .getOrCreate()
# spark.sparkContext.setLogLevel('FATAL')
# print('Spark session started Successfully')

# start_date = datetime(2024, 7, 1)
# end_date = datetime(2024, 7, 1)

# adrequest_path = []
# adrequest_openrtb_path =[]
# dmd_info_path = []
# throttle_cookies_path = []
# liveintent_cookies_path = []

# adrequest_base_path = 's3a://cdp-silver-us-b24/snowflake/adrequest/year={}/month={}/day={}/*.snappy.parquet'
# adrequest_openrtb_base_path = 's3a://cdp-silver-us-b24/snowflake/adrequest_openrtb/year={}/month={}/day={}/*.snappy.parquet'
# throttle_cookies_base_path = 's3a://cdp-silver-us-b24/throttle/cookies/year={}/month={}/day={}/*.snappy.parquet'
# dmd_info_base_path = 's3a://cdp-silver-us-b24/dmd-info/year={}/month={}/day={}/*.snappy.parquet'
# liveintent_cookies_base_path = 's3a://cdp-silver-us-b24/liveintent/us/year={}/month={}/day={}/*.snappy.parquet'
# today_date = start_date
# while today_date<=end_date:
#     year_str = str(today_date.year).zfill(4)
#     month_str = str(today_date.month).zfill(2)
#     day_str = str(today_date.day).zfill(2)

#     # Assign formatted paths to the respective variables and then append them to lists
#     adrequest_path.append(adrequest_base_path.format(year_str, month_str, day_str))
#     adrequest_openrtb_path.append(adrequest_openrtb_base_path.format(year_str, month_str, day_str))
#     throttle_cookies_path.append(throttle_cookies_base_path.format(year_str, month_str, day_str))
#     dmd_info_path.append(dmd_info_base_path.format(year_str, month_str, day_str))
#     liveintent_cookies_path.append(liveintent_cookies_base_path.format(year_str, month_str, day_str))

#     today_date += timedelta(days=1)

# snowflake_options = {
#     "sfURL": "ip85260.us-east-2.aws.snowflakecomputing.com",
#     "sfUser": "rudder",
#     "sfPassword": "Changeit@123",
#     "sfDatabase": "RUDDERDB",
#     "sfSchema": "RS_PROFILES_PROD",
#     "sfWarehouse": "RUDDERSTACK",
#     "sfRole": "RUDDER_RW"
# }
# non_endemic_publishers = ['62973efaa7a8ed16cb55bbfc', '6203b77823124f8d6c641973', '642d68a7543dac8932f559af']
# # curr_date = datetime.now()- timedelta(days = 23)
# # year = str(curr_date.year).zfill(4)
# # month = str(curr_date.month).zfill(2)
# # day = str(curr_date.day).zfill(2)

# # # #columns to fetch 
# adrequest_columns = ['docereecookieplatformuid', 'userplatformuid', 'platformuid', 'publisher_id', 'platformtype', 'physicianzone','zone']
# adrequest_openrtb_columns = ['docereecookieplatformuid', 'userplatformuid', 'platformuid', 'publisher_id', 'platformtype', 'physicianzone','zone']
# dmd_info_columns = ['platformuid']
# # # #paths for all cookies
# # adrequest_path = f's3a://cdp-silver-us-b24/snowflake/adrequest/year={year}/month={month}/day={day}/*.snappy.parquet'
# # adrequest_openrtb_path = f's3a://cdp-silver-us-b24/snowflake/adrequest_openrtb/year={year}/month={month}/day={day}/*.snappy.parquet'
# # # dmd_info_path = f's3a://cdp-silver-us-b24/dmd-info/year={year}/month={month}/day={day}/*.snappy.parquet'

# # # adrequest_df = spark.read.format('parquet').load(adrequest_path).select(adrequest_columns)
# # #adrequest_openrtb_df = spark.read.format('parquet').load(adrequest_openrtb_path).select(adrequest_openrtb_columns)

# try:
#     filtered_ad_request = spark.read.format("parquet") \
#         .option("header", "true") \
#         .option('inferSchema', 'true') \
#         .load(adrequest_path) \
#         .select(adrequest_columns) \
#         .filter(
#             (coalesce('physicianzone','zone')==1) & 
#             (
#                 ((col("docereecookieplatformuid").isNotNull()) & (length(col("docereecookieplatformuid")) > 0) & (col("docereecookieplatformuid") != '')) |
#                 ((col("userplatformuid").isNotNull()) & (length(col("userplatformuid")) > 0) & (col("userplatformuid") != '')) |
#                 ((col("platformuid").isNotNull()) & (length(col("platformuid")) > 0) & (col("platformuid") != ''))
#             ) &
#             ((col('publisher_id')!='') | (col('platformtype')!='') & (col('platformtype').isNotNull()) | (col('publisher_id').isNotNull()))
#         ) \
#         .dropDuplicates()
#     filtered_ad_request.filter((col('publisher_id').isNull()) & (col('platformtype').isNull())).show(truncate = False)
# #  filtered_ad_request.show(truncate = False)
# except Exception as e:
#     filtered_ad_request = None
#     print(e)

# try:
#     opentrb_filtered_ad_request = spark.read.format("parquet") \
#         .option("header", "true") \
#         .option('inferSchema', 'true') \
#         .load(adrequest_openrtb_path) \
#         .select(adrequest_openrtb_columns) \
#         .filter(
#         ((coalesce('physicianzone','zone')==1)) & 
#         (
#             ((col("userplatformuid").isNotNull()) & (length(col("userplatformuid")) > 0) & (col("userplatformuid") != ''))
#         ) &
#         ((col('publisher_id')!='') | (col('platformtype')!='') & (col('platformtype').isNotNull()) | (col('publisher_id').isNotNull()))
#     ) \
#     .dropDuplicates()
#     filtered_ad_request.filter((col('publisher_id').isNull()) & (col('platformtype').isNull())).show(truncate = False)
#     # opentrb_filtered_ad_request.filter((col('physicianzone') != '1') & (col('zone') != '1')).show(truncate = False)
# except Exception as e:
#     opentrb_filtered_ad_request = None

# # Process data to create a single 'cookie_id' column
# if filtered_ad_request is not None:
#     df1 = filtered_ad_request.select(col("docereecookieplatformuid").alias("cookie_id"), 
#                                     *[c for c in filtered_ad_request.columns if c not in ['docereecookieplatformuid', 'userplatformuid', 'platformuid']]) \
#                             .filter(col("docereecookieplatformuid").isNotNull()).distinct()
#     df2 = filtered_ad_request.select(col("userplatformuid").alias("cookie_id"), 
#                                     *[c for c in filtered_ad_request.columns if c not in ['docereecookieplatformuid', 'userplatformuid', 'platformuid']]) \
#                             .filter(col("userplatformuid").isNotNull()).distinct()
#     df3 = filtered_ad_request.select(col("platformuid").alias("cookie_id"), 
#                                     *[c for c in filtered_ad_request.columns if c not in ['docereecookieplatformuid', 'userplatformuid', 'platformuid']]) \
#                             .filter(col("platformuid").isNotNull()).distinct()
#     filtered_ad_request = df1.unionByName(df2).unionByName(df3).distinct().drop('physicianzone','zone')
#     # filtered_ad_request.show(truncate = False)

# if opentrb_filtered_ad_request is not None:
#     df4 = opentrb_filtered_ad_request.select(col("docereecookieplatformuid").alias("cookie_id"), 
#                                             *[c for c in opentrb_filtered_ad_request.columns if c not in ['docereecookieplatformuid', 'userplatformuid', 'platformuid']]) \
#                                     .filter(col("docereecookieplatformuid").isNotNull()).distinct()
#     df5 = opentrb_filtered_ad_request.select(col("userplatformuid").alias("cookie_id"), 
#                                             *[c for c in opentrb_filtered_ad_request.columns if c not in ['docereecookieplatformuid', 'userplatformuid', 'platformuid']]) \
#                                     .filter(col("userplatformuid").isNotNull()).distinct()
#     df6 = opentrb_filtered_ad_request.select(col("platformuid").alias("cookie_id"), 
#                                             *[c for c in opentrb_filtered_ad_request.columns if c not in ['docereecookieplatformuid', 'userplatformuid', 'platformuid']]) \
#                                     .filter(col("platformuid").isNotNull()).distinct()
#     opentrb_filtered_ad_request = df4.unionByName(df5).unionByName(df6).distinct().drop('physicianzone','zone')
#     # opentrb_filtered_ad_request.show(truncate = False)

# total_cookies_df = filtered_ad_request.unionByName(opentrb_filtered_ad_request).distinct()
# print(total_cookies_df.count())
# total_cookies_df.show(truncate = False)
# total_cookies_df.filter(col('publisher_id').isNull()).show(truncate = False)

# total_cookies_df = total_cookies_df.withColumn(
#         "network_type",
#         when(col("publisher_id").isin(non_endemic_publishers), 'non-endemic') 
#         .when(col("platformtype").isin('Tele-medicine Platform', 'E-Prescribing', 'Electronic Health Record'), 'poc') 
#         .when(col("platformtype").isin('Online Medical Association', 'Online Learning Portal', 'Medical News Platform', 'Physician Networking Platform', 'Online Medical Journal'), 'endemic')
#     ).drop('publisher_id', 'platformtype').distinct()
# total_cookies_df.filter(col('network_type').isNull()).show(truncate = False)
# # grouped_df = total_cookies_df.groupBy('cookie_id').agg(collect_set('network_type').alias('network_types')).filter(size(col('network_types')) > 1)
# # grouped_df.show(truncate = False)
# # print(grouped_df.count())

# # total_cookies_df.show(truncate = False)
# # print(f'Total cookies for {year}-{month}-{day} is :', total_cookies_df.count())
# total_cookies_count = total_cookies_df.count()
# total_unique_cookies_count = total_cookies_df.select('cookie_id').distinct().count()
# print(f'Total distinct cookies and network_type from {start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")} is :',total_cookies_count )
# print(f'Total unique cookies from {start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")} is :',total_unique_cookies_count)

# try:
#     query = f"SELECT distinct cookie_id FROM cookie_vs_npi"
#     cookies_after_stitching_df = spark.read.format("snowflake") \
#         .option("header", "true") \
#         .options(**snowflake_options) \
#         .option("query", query) \
#         .load()
#     # cookies_after_stitching_df.show(truncate  = False)
#     count = cookies_after_stitching_df.count()
#     logger.info(f'cookies after stitching till {(datetime.now() - timedelta(days = 2)).strftime("%Y-%m-%d")} is : {count}')
# except Exception as e:
#     logger.error(f'Error fetching data for {year}-{month}-{day}: {e}')

# # throttle_cookies_path = f's3a://cdp-silver-us-b24/throttle/cookies/year={year}/month={month}/day={day}/*.snappy.parquet'
# throttle_cookies_df = spark.read.format('parquet').load(throttle_cookies_path).select('doceree_id').distinct()
# # # throttle_cookies_df.show(truncate  = False)
# # print(f'cookies from throttle on {year}-{month}-{day} is :', throttle_cookies_df.count())
# print(f'cookies from throttle from {start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")} is :', throttle_cookies_df.count())

# # try:
# #     liveintent_cookies_path = f's3a://cdp-silver-us-b24/liveintent/us/year={year}/month={month}/day={day}/*.snappy.parquet'
# liveintent_cookies_df = spark.read.format('parquet').load(liveintent_cookies_path).select('single_doceree_id').distinct()
# # except:
# #     liveintent_cookies_df = None
# # print(f'cookies from liveintent on {year}-{month}-{day} is :', liveintent_cookies_df.count())
# print(f'cookies from liveintent from {start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")} is :', liveintent_cookies_df.count())

# # dmd_info_path = f's3a://cdp-silver-us-b24/dmd-info/year={year}/month={month}/day={day}/*.snappy.parquet'
# dmd_info_df =  spark.read.format('parquet').load(dmd_info_path).select('platformuid').distinct()
# # # dmd_info_df.show(truncate = False)
# # print(f'cookies from dmd_info on {year}-{month}-{day} is :', dmd_info_df.count())
# print(f'cookies from dmd_info {start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")} is :', dmd_info_df.count())

# cookie_match_from_throttle = cookies_after_stitching_df.join(throttle_cookies_df, cookies_after_stitching_df.COOKIE_ID==throttle_cookies_df.doceree_id, "inner").select(cookies_after_stitching_df.COOKIE_ID).distinct()
# # print(f'Cookies resolved from throttle on {year}-{month}-{day} is: ', cookie_match_from_throttle.count())
# print(f'Cookies resolved from throttle from {start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")} is: ', cookie_match_from_throttle.count())

# cookie_match_from_dmd = cookies_after_stitching_df.join(dmd_info_df, cookies_after_stitching_df.COOKIE_ID == dmd_info_df.platformuid, "inner").select(cookies_after_stitching_df.COOKIE_ID).distinct()
# # print(f'Cookies resolved from dmd_info on {year}-{month}-{day} is: ', cookie_match_from_dmd.count())
# print(f'Cookies resolved  from dmd_info from {start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")} is: ', cookie_match_from_dmd.count())

# cookie_match_from_liveintent = cookies_after_stitching_df.join(liveintent_cookies_df, cookies_after_stitching_df.COOKIE_ID == liveintent_cookies_df.single_doceree_id, "inner").select(cookies_after_stitching_df.COOKIE_ID).distinct()
# # print(f'Cookies resolved from liveintent on {year}-{month}-{day} is: ', cookie_match_from_liveintent.count())
# print(f'Cookies resolved from liveintent from {start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")} is: ', cookie_match_from_liveintent.count())

# total_cookies_resolved = cookie_match_from_throttle.union(cookie_match_from_liveintent).union(cookie_match_from_dmd).distinct()
# total_cookies_resolved_count = total_cookies_resolved.count()
# print(f'Total unique cookies resolved from {start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")} is',total_cookies_resolved_count)

# endemic_df = total_cookies_df.filter(col('network_type')=='endemic')
# endemic_count = endemic_df.count()
# # print(f'Endemic count for {year}-{month}-{day} is: ', endemic_count)
# print(f'Endemic count from {start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")} is: ', endemic_count)

# poc_df = total_cookies_df.filter(col('network_type')=='poc')
# poc_count = poc_df.count()
# # print(f'POC count for {year}-{month}-{day} is: ',poc_count)
# print(f'POC count from {start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")} is: ',poc_count)

# non_endemic_df = total_cookies_df.filter(col('network_type')=='non-endemic')
# non_endemic_count = non_endemic_df.count()
# # print(f'Non-Endemic count for {year}-{month}-{day} is: ', non_endemic_count)
# print(f'Non-Endemic count from {start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")} is: ', non_endemic_count)


# total_cookies_resolved_with_network_type = total_cookies_resolved.join(total_cookies_df, total_cookies_resolved.COOKIE_ID==total_cookies_df.cookie_id, 'left').select(
#     total_cookies_resolved.COOKIE_ID.alias('cookie_id'),
#     total_cookies_df.network_type
# )
# nullfilter = total_cookies_resolved_with_network_type.filter(col('network_type').isNull())
# print(nullfilter.count())
# notnullfilter = total_cookies_resolved_with_network_type.filter(col('network_type').isNotNull())
# print(notnullfilter.count())
# total_cookies_resolved_with_network_type.show(truncate = False)
# print('count of total cookies with network type is :', total_cookies_resolved_with_network_type.count())


# # endemic_resolved_df = endemic_df.join(
# #     total_cookies_resolved, endemic_df.cookie_id == total_cookies_resolved.COOKIE_ID, 'inner'
# # ).select(total_cookies_resolved.COOKIE_ID).distinct()
# # endemic_resolved_count = endemic_resolved_df.count()
# # # print(f'Resolved endemic count for {year}-{month}-{day} is: ', endemic_resolved_count)
# # print(f'Resolved endemic count from {start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")} is: ', endemic_resolved_count)

# # poc_resolved_df = poc_df.join(
# #     total_cookies_resolved, poc_df.cookie_id == total_cookies_resolved.COOKIE_ID, 'inner'
# # ).select(total_cookies_resolved.COOKIE_ID).distinct()
# # poc_resolved_count = poc_resolved_df.count()
# # # print(f'Resolved endemic count for {year}-{month}-{day} is: ', endemic_resolved_count)
# # print(f'Resolved poc count from {start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")} is: ', poc_resolved_count)


# # non_endemic_resolved_df = non_endemic_df.join(
# #     total_cookies_resolved, non_endemic_df.cookie_id == total_cookies_resolved.COOKIE_ID, 'inner'
# # ).select(total_cookies_resolved.COOKIE_ID).distinct()
# # non_endemic_resolved_count = non_endemic_resolved_df.count()
# # # print(f'Resolved endemic count for {year}-{month}-{day} is: ', endemic_resolved_count)
# # print(f'Resolved non_endemic count from {start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")} is: ', non_endemic_resolved_count)
# # #For Endemic ->for each partner(liveinetent, throttle, dmd_info)













# # # # # For endemic
# # endemic_resolved_df = total_cookies_df.filter(col('network_type') == 'endemic').join(
# #     cookies_after_stitching_df, total_cookies_df.cookie_id == cookies_after_stitching_df.COOKIE_ID, 'inner'
# # ).select(total_cookies_df.cookie_id).distinct()
# # endemic_resolved_count = endemic_resolved_df.count()
# # # print(f'Resolved endemic count for {year}-{month}-{day} is: ', endemic_resolved_count)
# # print(f'Resolved endemic count from {start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")} is: ', endemic_resolved_count)

# # # # endemic_cookie_match_from_throttle = endemic_resolved_df.join(throttle_cookies_df, endemic_resolved_df.COOKIE_ID==throttle_cookies_df.doceree_id, "inner").select(throttle_cookies_df.doceree_id).distinct()
# # # # print(f'Endemic Cookies resolved from throttle on {year}-{month}-{day} is: ', endemic_cookie_match_from_throttle.count())

# # # # endemic_cookie_match_from_dmd = endemic_resolved_df.join(dmd_info_df, endemic_resolved_df.COOKIE_ID == dmd_info_df.platformuid, "inner").select(dmd_info_df.platformuid).distinct()
# # # # print(f'Endemic Cookies resolved from dmd_info on {year}-{month}-{day} is: ', endemic_cookie_match_from_dmd.count())
# # # # # For poc
# # poc_resolved_df = total_cookies_df.filter(col('network_type') == 'poc').join(
# #     cookies_after_stitching_df, total_cookies_df.cookie_id == cookies_after_stitching_df.COOKIE_ID, 'inner'
# # ).select(total_cookies_df.cookie_id).distinct()
# # poc_resolved_count = poc_resolved_df.count()
# # # print(f'Resolved POC count for {year}-{month}-{day} is: ', poc_resolved_count)
# # print(f'Resolved POC count from {start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")} is: ', poc_resolved_count)

# # # # poc_cookie_match_from_throttle = poc_resolved_df.join(throttle_cookies_df, poc_resolved_df.COOKIE_ID==throttle_cookies_df.doceree_id, "inner").select(throttle_cookies_df.doceree_id).distinct()
# # # # print(f'POC Cookies resolved from throttle on {year}-{month}-{day} is: ', poc_cookie_match_from_throttle.count())

# # # # poc_cookie_match_from_dmd = poc_resolved_df.join(dmd_info_df, poc_resolved_df.COOKIE_ID == dmd_info_df.platformuid, "inner").select(dmd_info_df.platformuid).distinct()
# # # # print(f'POC resolved from dmd_info on {year}-{month}-{day} is: ', poc_cookie_match_from_dmd.count())

# # # # # For non-endemic
# # non_endemic_resolved_df = total_cookies_df.filter(col('network_type') == 'non-endemic').join(
# #     cookies_after_stitching_df, total_cookies_df.cookie_id == cookies_after_stitching_df.COOKIE_ID, 'inner'
# # ).select(total_cookies_df.cookie_id).distinct()
# # non_endemic_resolved_count = non_endemic_resolved_df.count()
# # # print(f'Resolved non-endemic count for {year}-{month}-{day} is: ', non_endemic_resolved_count)
# # print(f'Resolved non-endemic count  from {start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")} is: ', non_endemic_resolved_count)

# # # # non_endemic_cookie_match_from_throttle = non_endemic_resolved_df.join(throttle_cookies_df, non_endemic_resolved_df.COOKIE_ID==throttle_cookies_df.doceree_id, "inner").select(throttle_cookies_df.doceree_id).distinct()
# # # # print(f'Non-Endemic Cookies resolved from throttle on {year}-{month}-{day} is: ', endemic_cookie_match_from_throttle.count())

# # # # endemic_cookie_match_from_dmd = non_endemic_resolved_df.join(dmd_info_df, non_endemic_resolved_df.COOKIE_ID == dmd_info_df.platformuid, "inner").select(dmd_info_df.platformuid).distinct()
# # # # print(f'Non-Endemic Cookies resolved from dmd_info on {year}-{month}-{day} is: ', endemic_cookie_match_from_dmd.count())

# total_cookies_resolved_precentage = (total_cookies_resolved_count/ total_unique_cookies_count * 100) if total_unique_cookies_count > 0 else 0
# # endemic_resolved_percentage = (endemic_resolved_count / endemic_count * 100) if endemic_count > 0 else 0
# # poc_resolved_percentage = (poc_resolved_count / poc_count * 100) if poc_count > 0 else 0
# # non_endemic_resolved_percentage = (non_endemic_resolved_count / non_endemic_count * 100) if non_endemic_count > 0 else 0


# september_metrics = Row(
#     year=year_str,
#     month=7, 
#     total_cookies=total_unique_cookies_count,
#     total_cookies_resolved_count = total_cookies_resolved_count,
#     total_cookies_resolved_precentage = total_cookies_resolved_precentage,
#     # endemic_count = endemic_count,
#     # endemic_resolved_count=endemic_resolved_count,
#     # poc_count = poc_count,
#     # poc_resolved_count=poc_resolved_count,
#     # non_endemic_count = non_endemic_count,
#     # non_endemic_resolved_count=non_endemic_resolved_count,
#     # endemic_resolved_percentage=endemic_resolved_percentage,
#     # poc_resolved_percentage=poc_resolved_percentage,
#     # non_endemic_resolved_percentage=non_endemic_resolved_percentage,
#     updated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
# )
# metrics_df = spark.createDataFrame([september_metrics])
# metrics_df.show(truncate= False)

# # metrics_df.write \
# #     .format("snowflake") \
# #     .options(**snowflake_options) \
# #     .option("dbtable", "data_metrics") \
# #     .mode("append") \
# #     .save()

# # # Write data to Snowflake
# # df.write \
# #     .format("net.snowflake.spark.snowflake") \
# #     .options(**sf_options) \
# #     .option("dbtable", "NPI_AM_DATA") \
# #     .mode("overwrite") \
# #     .save()



# from pyspark.sql import SparkSession, Row
# from datetime import datetime, timedelta
# import logging 
# from pyspark.sql.functions import *
# from calendar import monthrange

# logger = logging.getLogger()
# logging.basicConfig(level=logging.INFO)
# # Initialize Spark session
# spark = SparkSession.builder \
#     .appName("Daily Refresh Count") \
#     .config("spark.hadoop.fs.s3a.access.key", "") \
#     .config("spark.hadoop.fs.s3a.secret.key", "") \
#     .getOrCreate()
# spark.sparkContext.setLogLevel('FATAL')
# print('Spark session started Successfully')

# liveintent_columns = ['liveintent_non_id', 'dmd_id']

# liveintent_cookies_base_path = 's3a://cdp-raw-us-b24/liveintent/us'
# dmd_feed_path = 's3a://cdp-silver-us-b24/dmd-feed/2024/11/03/*.snappy.parquet'


# def read_data(base_path, columns, start_date, end_date):
#     # start_date = datetime(2024,1,1)
#     # end_date = datetime(2024,1,4)
#     def align_schemas(dfs):
#         # Find all columns that appear in any dataframe
#         all_columns = set()
#         for df in dfs:
#             all_columns.update(df.columns)
        
#         # Add missing columns with null values
#         for i in range(len(dfs)):
#             for column in all_columns - set(dfs[i].columns):
#                 dfs[i] = dfs[i].withColumn(column, lit(None))
#             dfs[i] = dfs[i].select(sorted(dfs[i].columns))
        
#         return dfs

#     def read_parquet(paths):
#         return spark.read.format("csv")\
#                 .option("inferSchema", "false")\
#                 .option("header", "true").load(paths).select(columns).distinct()
    
#     start = start_date
#     end = end_date
#     all_dfs = []

#     while start <= end:
#         year = str(start.year).zfill(4)
#         month = str(start.month).zfill(2)
#         day = str(start.day).zfill(2)

#         try:
#             path = f"{base_path}/{year}/{month}/{day}/*.csv"
#             df = read_parquet(path)
#             all_dfs.append(df)
#         except Exception as e:
#             print(f"Failed to read data for {path}: {e}")
#             pass
        
#         start += timedelta(days=1)

#     # Align schemas of all dataframes
#     aligned_dfs = align_schemas(all_dfs)

#     # Union all dataframes
#     final_df = aligned_dfs[0]
#     for df in aligned_dfs[1:]:
#         final_df = final_df.unionByName(df)
    
#     return final_df

# for month in range(1,2):
#     num_days_in_month = monthrange(2024, month)[1]
#     start_date = datetime(2024, month, 1)
#     end_date = datetime(2024, month, 1)

#     liveintent_cookies_df = read_data(liveintent_cookies_base_path, liveintent_columns, start_date ,end_date).distinct()
#     print(liveintent_cookies_df.count())
#     groupedDF = (
#                 liveintent_cookies_df
#                 .groupBy('liveintent_non_id')
#                 .agg(count('*').alias('count'),
#                      collect_set('dmd_id').alias('dmd_ids'),)
#                 .orderBy('count', ascending=False)
#                 )   
#     groupedDF.show(truncate = False)

#     # liveintent_cookies_df.show(truncate = False)
#     dmd_feed_df = spark.read.format('parquet').load(dmd_feed_path).select('dgid','npi_number')
#     print(dmd_feed_df.count())
#     # dmd_feed_df.show(truncate = False)

#     liveintent_id_to_npi_df = liveintent_cookies_df.join(dmd_feed_df, liveintent_cookies_df.dmd_id == dmd_feed_df.dgid, 'inner').select(liveintent_cookies_df.liveintent_non_id, dmd_feed_df.npi_number)
#     print(liveintent_id_to_npi_df.count())
#     liveintent_id_to_npi_df.show(truncate = False)
#     month_str = str(month).zfill(2)
#     dest_path = f's3a://cdp-raw-us-b24/liveintent_npi/2024/{month_str}/'
#     # liveintent_id_to_npi_df.write.option("header", "true").option("quoteAll", "true").option("escape", "\"").mode('overwrite').parquet(dest_path)

####################################################################################################################

# ####feature count for hcp_id
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import requests
import pygeohash
from pyspark.sql.types import StringType,LongType
from user_agents import parse
from ua_parser import user_agent_parser
import ipaddress

# .config("spark.driver.maxResultSize", "2g")\
spark = SparkSession.builder \
        .appName("Feature Count for HCP_ID") \
        .config("spark.hadoop.fs.s3a.access.key", "") \
        .config("spark.hadoop.fs.s3a.secret.key", "") \
        .config("spark.hadoop.fs.s3a.endpoint", "s3-us-east-2.amazonaws.com")\
        .getOrCreate()

spark.sparkContext.setLogLevel('FATAL')
spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")

sf_options_4 = {
    "sfURL": "ip85260.us-east-2.aws.snowflakecomputing.com",
    "sfUser": "rudder",
    "sfPassword": "Changeit@123",
    "sfDatabase": "RUDDERDB",
    "sfSchema": "RS_PROFILES_DS",
    "sfWarehouse": "RUDDERSTACK",
    "sfRole": "RUDDER_RW"
}

sf_options = {
    "sfURL": "ip85260.us-east-2.aws.snowflakecomputing.com",
    "sfUser": "rudder",
    "sfPassword": "Changeit@123",
    "sfDatabase": "DOCEREE_ANALYTICS",
    "sfSchema": "PLATFORM",
    "sfWarehouse": "RUDDERSTACK",
    "sfRole": "RUDDER_RW"
}

sf_options_2 = {
    "sfURL": "ip85260.us-east-2.aws.snowflakecomputing.com",
    "sfUser": "rudder",
    "sfPassword": "Changeit@123",
    "sfDatabase": "QA_DOCEREE_ANALYTICS",
    "sfSchema": "PLATFORM",
    "sfWarehouse": "RUDDERSTACK",
    "sfRole": "RUDDER_RW"
}

sf_options_3 = {
    "sfURL": "ip85260.us-east-2.aws.snowflakecomputing.com",
    "sfUser": "rudder",
    "sfPassword": "Changeit@123",
    "sfDatabase": "RUDDERDB",
    "sfSchema": "RS_PROFILES_PROD",
    "sfWarehouse": "RUDDERSTACK",
    "sfRole": "RUDDER_RW"
}

print('Spark session started successfully.')

def parse_useragent(user_agent_string):
    # Default replacement for missing values
    default_value = "*"
    
    # Parse the User-Agent using `user_agents` library
    try:
        user_agent = parse(user_agent_string or "")
        
        # Operating System
        os_info = f"{user_agent.os.family}".strip() or default_value
        
        # Browser (family only, without version)
        browser_info = f"{user_agent.browser.family}".strip() or default_value
        
        # Device Type
        if user_agent.is_mobile:
            device_type = "Mobile"
        elif user_agent.is_tablet:
            device_type = "Tablet"
        elif user_agent.is_pc:
            device_type = "PC"
        elif user_agent.is_touch_capable:
            device_type = "Touch"
        elif user_agent.is_bot:
            device_type = "Bot"
        else:
            device_type = default_value
        
        # Combine the extracted information with `|` as delimiter
        result_string = "|".join([os_info, browser_info, device_type])
        return result_string
    except Exception as e:
        # In case of any errors, return default values
        return None

def ip_to_integer(ip):
    try:
        return int(ipaddress.ip_address(ip))
    except ValueError:
        return None  # Handle invalid IP addresses

# and coalesce(length(usernpi), length(userhashednpi)) > 5
def process_data():
    try:
        adrequest_query = f"""select usernpi, 
        userhashednpi, 
        useremail, 
        userhashedemail,
        useragent, 
        bidrequestip, 
        created_at, 
        HH, 
        userplatformuid as cookie_id, 
        publisherdomain, 
        publisherrequestedurl, 
        platformtype,
        usercity, 
        userstate, 
        userzipcode, 
        usergender, 
        userspecialization  
        from doceree_analytics.platform.ad_request_openrtb 
        where date(created_at) >= '2024-10-01' 
        and date(created_at) <= '2024-11-30' 
        and publisher_id = '64e83ecd97184e7542471443' 
        and coalesce(physicianzone, zone) = 1 
        union
        select usernpi, 
            userhashednpi, 
            useremail, 
            userhashedemail,
            useragent, 
            bidrequestip, 
            created_at, 
            HH, 
            docereecookieplatformuid as cookie_id, 
            publisherdomain, 
            publisherrequestedurl, 
            platformtype,
            usercity, 
            userstate, 
            userzipcode, 
            usergender, 
            userspecialization  
            from doceree_analytics.platform.ad_request 
            where date(created_at) >= '2024-10-01' 
            and date(created_at) <= '2024-11-30' 
            and coalesce(physicianzone, zone) = 1 """
         
        hashed_npi_query = 'select * from hashed_npi_data'

        ip_query = """select * from NET_ACUITY_DMP"""

        cookie_to_npi_query = """SELECT 
                cookie_id::STRING AS cookie,
                npi_array[0]::STRING AS npi
            FROM (
                SELECT 
                    cookie_id, 
                    array_agg(npi) AS npi_array,
                    COUNT(npi) AS npi_count
                FROM 
                    cookie_vs_npi 
                GROUP BY 
                    cookie_id
                ORDER BY 
                    npi_count DESC
            ) subquery
            WHERE 
                npi_count = 1
            """
        cookie_to_npi_df = spark.read.format("snowflake") \
            .option("header", "true") \
            .option("quote", "\"") \
            .option("escape", "\"") \
            .option("multiline", "true") \
            .options(**sf_options_3) \
            .option("query", cookie_to_npi_query) \
            .load()
        cookie_to_npi_df = cookie_to_npi_df.withColumn('NPI',col('NPI').cast('long'))
        # cookie_to_npi_df.show(truncate = False)

        ip_df = spark.read.format("snowflake") \
            .option("header", "true") \
            .option("quote", "\"") \
            .option("escape", "\"") \
            .option("multiline", "true") \
            .options(**sf_options_2) \
            .option("query", ip_query) \
            .load()
        # # print('No of rows in NET_ACUITY_DMP are: ',ip_df.count())
        # # ip_df.show(truncate = False)
        
        adrequest_df = spark.read.format("snowflake") \
            .option("header", "true") \
            .option("quote", "\"") \
            .option("escape", "\"") \
            .option("multiline", "true") \
            .options(**sf_options) \
            .option("query", adrequest_query) \
            .load()
        # print('No of rows for adrequest are: ', adrequest_df.count())
        # adrequest_df.show(truncate = False)

        hashed_npi_df = spark.read.format("snowflake") \
            .option("header", "true") \
            .option("quote", "\"") \
            .option("escape", "\"") \
            .option("multiline", "true") \
            .options(**sf_options) \
            .option("query", hashed_npi_query) \
            .load()
        # print('No of rows for hashed_npi are: ', hashed_npi_df.count())
        # hashed_npi_df.show(truncate = False)
        # print(adrequest_df.count())
        npi_df = (
                adrequest_df
                .join(hashed_npi_df, adrequest_df.USERHASHEDNPI == hashed_npi_df.HASHED_NPI, 'left')
                .withColumn(
                    'HCP_ID', 
                    when(length(col('USERNPI')) == 10, col('USERNPI'))
                    .otherwise(col('NPI'))
                )
            )
        # resolved_npi_from_hashed_npi = npi_df.filter(col('HCP_ID').isNotNull())
        # print(resolved_npi_from_hashed_npi.count())
        # print(npi_df.count())
        # npi_df.show(truncate = False)
        npi_df = (
            npi_df
            .join(cookie_to_npi_df, npi_df.COOKIE_ID == cookie_to_npi_df.COOKIE, 'left')
            .withColumn(
                'HCP_ID', 
                when(col('HCP_ID').isNotNull(), col('HCP_ID'))
                .otherwise(cookie_to_npi_df.NPI)
            )
        )
        # resolved_npi_after_cookie = npi_df.filter(col('HCP_ID').isNotNull())
        # print(resolved_npi_after_cookie.count())
        # print(npi_df.count())
        # npi_df.show(truncate = False)
        npi_df =  npi_df.filter(col('HCP_ID').isNotNull())
        # print(npi_df.count())
        npi_df = npi_df.withColumn('HCP_ID', col('HCP_ID').cast('long'))

        # Select all columns from adrequest_df except 'USERNPI' and 'USERHASHEDNPI', plus the new HCP_ID column
        columns_to_select = [col for col in adrequest_df.columns if col not in ('USERNPI', 'USERHASHEDNPI')]
        final_columns = ['HCP_ID']+ columns_to_select

        # Select the final columns
        npi_df = npi_df.select(*final_columns)
        # count_npi = npi_df.filter(length(col('HCP_ID')) != 10).collect()
        # print(len(count_npi))
        # npi_df.filter(length(col('HCP_ID')) != 10).show(truncate = False)
        # npi_df.filter(col('HCP_ID').isNull()).show(truncate = False)
        # print(npi_df.count())
    
        npi_df = npi_df.withColumn(
            'hour',
            lpad(col('HH').cast('string'), 2, '0')
        )
        npi_df = npi_df.withColumn('day_of_week', lpad(date_format(col('created_at'), 'u'), 2, '0'))

        get_useragent_udf = udf(parse_useragent,StringType())

        npi_df = npi_df.withColumn('parsed_user_agent', get_useragent_udf(col('useragent')))

        hcp_id_counts_df = (
            npi_df.groupBy("HCP_ID")
            .agg(count("*").alias("adrequest_count"))
        )
        npi_df = (
            npi_df.join(hcp_id_counts_df, on="HCP_ID", how="left")
        )
        # npi_df.show(truncate =False)

        # npi_df.show(truncate = False)

        # ip_to_integer_udf = udf(ip_to_integer, LongType())
        # npi_df = npi_df.withColumn("bidrequestip_int", ip_to_integer_udf(col("bidrequestip")))
        # print(npi_df.count())
        # npi_df.show(truncate = False)
        # npi_df.filter(length(col('HCP_ID'))!=10).show(truncate = False)
        # npi_df.filter(col('HCP_ID').isNull()).show(truncate = False)
        # ip_df = ip_df.select('CITY_NAME', 'POSTAL_CODE', 'COUNTRY_NAME','SUBDIVISION_1_NAME', 'NETWORK_START_INT', 'NETWORK_END_INT').filter(col('COUNTRY_NAME')=='United States')
        # ip_df = ip_df.repartition(100)
        # ip_df.show(truncate = False)
        # print(ip_df.count())

        # from bisect import bisect_left

        # # Sort the ip_df by NETWORK_START_INT
        # ip_df_sorted = ip_df.orderBy("NETWORK_START_INT").collect()

        # # Define a function to find the range using binary search
        # def find_ip_range(ip_value, ip_df_sorted):
        #     # Extract the sorted start and end IPs from ip_df
        #     start_ips = [row['NETWORK_START_INT'] for row in ip_df_sorted]
        #     end_ips = [row['NETWORK_END_INT'] for row in ip_df_sorted]
            
        #     # Use binary search to find the relevant range
        #     idx = bisect_left(start_ips, ip_value)
            
        #     if idx < len(start_ips) and start_ips[idx] <= ip_value <= end_ips[idx]:
        #         return ip_df_sorted[idx]
        #     return None

        # # UDF to apply binary search to each row in npi_df
        # @udf("struct<ip_state:string, ip_country:string, ip_zipcode:string>")
        # def get_ip_info_from_range(ip_value):
        #     # Find the relevant IP range from the sorted ip_df
        #     match = find_ip_range(ip_value, ip_df_sorted)
            
        #     if match:
        #         return (match['ip_state'], match['ip_country'], match['ip_zipcode'])
        #     else:
        #         return (None, None, None)

        # npi_df = npi_df.withColumn(
        #     "ip_info",
        #     get_ip_info_from_range(col('bidrequestip_int'))
        # )
        # print(npi_df.count())
        # npi_df.show(truncate = False)

        # # Extract the fields from the ip_info column
        # npi_df = npi_df.withColumn('ip_state', npi_df['ip_info']['ip_state']) \
        #             .withColumn('ip_country', npi_df['ip_info']['ip_country']) \
        #             .withColumn('ip_zipcode', npi_df['ip_info']['ip_zipcode'])

        # result_df = npi_df.join(
        #     (ip_df),
        #     (npi_df["bidrequestip_int"] >= ip_df["NETWORK_START_INT"]) & 
        #     (npi_df["bidrequestip_int"] <= ip_df["NETWORK_END_INT"]),
        #     "left"
        # )
        
        # print(result_df.count())
        # result_df.show(truncate = False)


        # # Select required columns and add new location details
        # result_df = result_df.select(
        #     npi_df["*"],  # All original columns from npi_df
        #     ip_df["POSTAL_CODE"].alias("ip_zipcode"),
        #     ip_df["SUBDIVISION_1_NAME"].alias("ip_state"),
        #     ip_df["CITY_NAME"].alias("ip_city"),
        #     ip_df["COUNTRY_NAME"].alias("ip_country")
        # )

        # # Drop intermediate columns if necessary
        # result_df = result_df.drop("ip_int", "start_ip_int", "end_ip_int")
        # result_df.show(truncate = False)

        # npi_df.show(truncate = False)
        
        # # get_geohash_udf = udf(get_geohash_from_ip, StringType())

        # # # Add the 'geohash6' column to npi_df
        # # npi_df = npi_df.withColumn('geohash6', get_geohash_udf(col('bidrequestip')))


        # # print('No of rows with non null HCP_ID are:', npi_df.count())
        # # npi_df.printSchema()

        # # platformtype_counts_df = (
        # #     npi_df.groupBy("HCP_ID", "platformtype")  # Group by HCP_ID and platformtype
        # #     .agg(count("platformtype").alias("platformtype_count"))  # Count occurrences
        # # )
        # # platformtype_counts_df.show(truncate =  False)

        # # # Step 2: Aggregate the counts into a JSON-like structure for each HCP_ID
        # # aggregated_df = (
        # #     platformtype_counts_df.groupBy("HCP_ID")  # Group by HCP_ID again
        # #     .agg(
        # #         expr(
        # #             "map_from_entries(collect_list(named_struct('value', platformtype, 'count', platformtype_count)))"
        # #         ).alias("platformtype_counts")  # Create map for platformtype counts
        # #     )
        # # )
        columns_to_aggregate = [
            "platformtype",
            "cookie_id",
            "useremail",
            "userhashedemail",
            "useragent",
            'parsed_user_agent',
            "bidrequestip",
            "hour",
            "day_of_week",
            "publisherdomain",
            "publisherrequestedurl",
            'usercity', 
            'userstate', 
            'userzipcode',  
            'userspecialization',
            "adrequest"
            # 'geohash6',
             # 'usergender'
        ]

        # # # Initialize an empty list to store intermediate DataFrames
        # # aggregated_dfs = []

        # # # Loop through each column and create a DataFrame for counts
        # # for column in columns_to_aggregate:
        # #     cleaned_df = npi_df.filter(col(column).isNotNull() & (col(column) != ''))
        # #     # cleaned_df.show(truncate = False)
        # #     # Step 1: Count distinct values for each HCP_ID and the current column
        # #     temp_df = (
        # #         cleaned_df.groupBy("HCP_ID", column)
        # #         .agg(count(column).alias(f"{column}_count"))
        # #     )
        # #     # temp_df.show(truncate = False)
            
        # #     # Step 2: Aggregate counts into a JSON-like map structure
        # #     aggregated_column_df = (
        # #         temp_df.groupBy("HCP_ID")
        # #         .agg(
        # #             expr(
        # #                 f"map_from_entries(collect_list(named_struct('value', {column}, 'count', {column}_count)))"
        # #             ).alias(f"{column}_counts")
        # #         )
        # #     )
            
        # #     # Add the aggregated column DataFrame to the list
        # #     aggregated_dfs.append(aggregated_column_df)

        # # # Combine all the aggregated DataFrames
        # # final_aggregated_df = aggregated_dfs[0]
        # # for aggregated_df in aggregated_dfs[1:]:
        # #     # Join on HCP_ID to combine results
        # #     final_aggregated_df = final_aggregated_df.join(aggregated_df, on="HCP_ID", how="inner")


        # # # final_aggregated_df =  final_aggregated_df.orderBy('useremail_counts', ascending=False)
        # # final_aggregated_df.show(truncate = False)
        # # for column in final_aggregated_df.columns:
        # #     if 'counts' in column:  # Assuming all map columns have '_counts' in their name
        # #         final_aggregated_df = final_aggregated_df.withColumn(column, to_json(col(column)))

        # Initialize an empty list to store the aggregated DataFrames
        aggregated_dfs = []
        # print(npi_df.count())
        # Loop through each column and calculate value counts per HCP_ID
        for column in columns_to_aggregate:
            if column in ["cookie_id", "publisherrequestedurl", "publisherdomain", "useragent", 'userhashedemail','adrequest']:
                pass
            else:
                npi_df = npi_df.withColumn(column, lower(trim(col(column))))
            if column == "adrequest":
                # Special handling for adrequest_count
                value_counts_df = (
                    hcp_id_counts_df
                    .withColumn("column_value", lit("adrequest_count"))
                    .withColumnRenamed("adrequest_count", "value_count")
                    .withColumn("column_name", lit("adrequest"))
                )
                # value_counts_df.show(truncate = False)
            else:

                # Filter out null and empty values
                cleaned_df = npi_df.filter(col(column).isNotNull() & (col(column) != ''))
                # cleaned_df.show(truncate = False)
                
                # Group by HCP_ID and the column value, then count occurrences
                value_counts_df = (
                    cleaned_df.groupBy("HCP_ID", column)
                    .agg(count(column).alias("value_count"))
                    .withColumn("column_name", lit(column))  
                    .withColumnRenamed(column, "column_value")  
                )
                # value_counts_df.show(truncate = False)
                
                # Append to the list of DataFrames
            aggregated_dfs.append(value_counts_df)

        # Union all the individual DataFrames into a single DataFrame
        final_value_counts_df = aggregated_dfs[0]
        for df in aggregated_dfs[1:]:
            final_value_counts_df = final_value_counts_df.unionByName(df)

        # final_value_counts_df = (
        #     final_value_counts_df.join(hcp_id_counts_df, on="HCP_ID", how="left")
        #     .withColumnRenamed("hcp_id_count", "hcp_id_total_count")
        # )

        # final_value_counts_df.show(truncate=False)

        # Final output
        # print(final_value_counts_df.count())
        # final_value_counts_df.filter(col('hcp_id') == '1437223989').show(n=1000, truncate = False)

        # # # Write this DataFrame to S3
        # # dest_path = 's3a://cdp-raw-us-b24/prior_probability_computation1/'
        # # final_value_counts_df.coalesce(1).write.mode("overwrite").csv(dest_path, header=True)

        # Write to Snowflake
        final_value_counts_df.write \
            .format("net.snowflake.spark.snowflake") \
            .options(**sf_options_4) \
            .option("dbtable", "FREQUENCY_COUNT_LONG_1") \
            .mode("overwrite") \
            .save()

        # # # dest_path = 's3a://cdp-raw-us-b24/prior_probability_computation/'
        # # # final_aggregated_df.coalesce(1).write.mode('overwrite').csv(dest_path)

        # # # final_aggregated_df.write \
        # # #     .format("net.snowflake.spark.snowflake") \
        # # #     .options(**sf_options) \
        # # #     .option("dbtable", "FREQUENCY_COUNT") \
        # # #     .mode("overwrite") \
        # # #     .save()
        
    except Exception as e:
        print(e)

process_data()
# # ####################################################################################################################################################

# import requests
# import pygeohash



# # # Function to get geohash from IP
# def get_geohash_from_ip(ip_address):
#     """
#     Get geohash from an IP address.
    
#     Parameters:
#         ip_address (str): The IP address to geolocate.
    
#     Returns:
#         str: A geohash (precision 6) or None if lookup fails.
#     """
#     try:
#         url = f"http://ip-api.com/json/{ip_address}"
#         response = requests.get(url)
#         if response.status_code == 200:
#             data = response.json()
#             if data['status'] == 'success':
#                 latitude = data['lat']
#                 longitude = data['lon']
#                 return pygeohash.encode(latitude, longitude, precision=6)
#         return None  # Return None if API fails or IP is invalid
#     except Exception as e:
#         return None  # Handle any unexpected exceptions gracefully


# print(get_geohash_from_ip('69.139.54.10'))


# from pyspark.sql import SparkSession
# from pyspark.sql.functions import *
# import requests
# import pygeohash
# from pyspark.sql.types import StringType


# spark = SparkSession.builder \
#         .appName("Feature Count for HCP_ID") \
#         .config("spark.hadoop.fs.s3a.access.key", "") \
#         .config("spark.hadoop.fs.s3a.secret.key", "") \
#         .config("spark.hadoop.fs.s3a.endpoint", "s3-us-east-2.amazonaws.com")\
#         .getOrCreate()

# spark.sparkContext.setLogLevel('FATAL')
# spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")

# sf_options = {
#     "sfURL": "ip85260.us-east-2.aws.snowflakecomputing.com",
#     "sfUser": "rudder",
#     "sfPassword": "Changeit@123",
#     "sfDatabase": "RUDDERDB",
#     "sfSchema": "RS_PROFILES_PROD",
#     "sfWarehouse": "RUDDERSTACK",
#     "sfRole": "RUDDER_RW"
# }
# print('Spark session started successfully.')

# def sanitize_column_names(df):
#     for col in df.columns:
#         new_col_name = col.lower().strip().replace("(", "").replace(")", "").replace(" ", "_").replace('.', '')
#         df = df.withColumnRenamed(col, new_col_name)
#     return df

# def convert_to_spark_timestamp_for_date_columns(df):
#     df = (
#         df
#     .withColumn('Last Update Date', to_date(col('Last Update Date'), 'MM/dd/yyyy'))
#     .withColumn('NPI Deactivation Date', to_date(col('NPI Deactivation Date'), 'MM/dd/yyyy'))
#     .withColumn('NPI Reactivation Date', to_date(col('NPI Reactivation Date'), 'MM/dd/yyyy'))
#     .withColumn('Certification Date', to_date(col('Certification Date'), 'MM/dd/yyyy'))
#     .withColumn('Provider Enumeration Date', to_date(col('Provider Enumeration Date'), 'MM/dd/yyyy'))
#     )
#     return df

# # src_path = 's3a://cdp-raw-us-b24/medical-council-new/2024/12/02/*.csv'
# src_path = 's3a://cdp-raw-us-b24/medical-council-new/2024/12/02/medical_council_new/*.parquet'
# # df = spark.read.format('csv').option('header','true').option("quote", "\"")\
# #                     .option("escape", "\"")\
# #                     .option("multiline", "true").load(src_path)
# df = spark.read.format('parquet').option('header','true').load(src_path)
# # df.printSchema()
# # df = convert_to_spark_timestamp_for_date_columns(df)
# # df = sanitize_column_names(df)
# # df = df.withColumn('npi', col('npi').cast('long'))
# df.printSchema()
# # dest_path = 's3a://cdp-raw-us-b24/medical-council-new/2024/12/02/medical_council_new/'
# # df.coalesce(1).write.mode('overwrite').parquet(dest_path)
# df.write \
#     .format("net.snowflake.spark.snowflake") \
#     .options(**sf_options) \
#     .option("dbtable", "MEDICAL_COUNCIL") \
#     .option("quoteIdentifiers", "false") \
#     .mode("overwrite") \
#     .save()
# # print(f"Successfully processed and saved data for {year}-{month}-{day}")

# from user_agents import parse
# from ua_parser import user_agent_parser

# def parse_useragent(user_agent_string):
#     # Parse the User-Agent
#     user_agent = parse(user_agent_string)
 
#     user_agent
 
#     # Extract detailed information
#     print("Is a bot:", user_agent.is_bot)
#     print("Is mobile:", user_agent.is_mobile)
#     print("Is tablet:", user_agent.is_tablet)
#     print("Is touch-capable:", user_agent.is_touch_capable)
#     print("Is PC:", user_agent.is_pc)
#     print("Is email client:", user_agent.is_email_client)
 
#     # Operating system
#     print("Operating System:", user_agent.os.family, user_agent.os.version_string)
 
#     # Browser
#     print("Browser:", user_agent.browser.family, user_agent.browser.version_string)
 
#     # Device
#     print("Device:", user_agent.device.family, user_agent.device.brand, user_agent.device.model)
 
#     # Parse User-Agent
#     parsed_ua = user_agent_parser.Parse(user_agent_string)
 
#     # Print parsed information
#     print("Device:", parsed_ua['device'])
#     print("OS:", parsed_ua['os'])
#     print("User-Agent:", parsed_ua['user_agent'])

# parse_useragent('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36')

# import requests

# def get_location_with_poi(ip_address):
#     """
#     Get location details and nearby points of interest using IP address.
 
#     Parameters:
#         ip_address (str): The IP address to geolocate.
#         api_key (str): Your ipstack API key.
 
#     Returns:
#         dict: Location details including nearby POIs.
#     """
#     api_key = "116e3bb5cef2020bcea2d450dbc90b4a"
#     url = f"http://api.ipstack.com/{ip_address}?access_key={api_key}"
#     response = requests.get(url)
    
#     if response.status_code == 200:
#         data = response.json()
#         print("API Response:", data)  # Debug: Print the entire API response
        
#         if 'success' in data and data['success'] == False:
#             # Handle the error returned by ipstack
#             return {'error': data.get('error', {}).get('info', 'Unknown API error')}
        
#         # Extracting location details
#         location_details = {
#             'IP': ip_address,
#             'Country': data.get('country_name', 'Not available'),
#             'Region': data.get('region_name', 'Not available'),
#             'City': data.get('city', 'Not available'),
#             'Latitude': data.get('latitude', 'Not available'),
#             'Longitude': data.get('longitude', 'Not available'),
#             'ISP': data.get('connection', {}).get('isp', 'Not available'),
#             'POI': data.get('poi', 'Not available')  # Points of Interest if available
#         }
#         return location_details
#     else:
#         return {'error': f"HTTP Error {response.status_code}"}
    
# print(get_location_with_poi('69.139.54.10'))