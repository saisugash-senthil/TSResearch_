import pyodbc
server_name = 'SAILAPTOP\SQLEXPRESS'
database_name = 'DBMS'
username = 'dbo'
password = '1234'
parameters = []

# Connection string using format method
connection_string = 'DRIVER={{SQL Server}};SERVER={};DATABASE={};Trusted_Connection=yes'.format(server_name, database_name)
connection = pyodbc.connect(connection_string)
cursor = connection.cursor()

cursor.execute('select * from time_series')
data = cursor.fetchall()

for row in data:
    parameters.append(row[0])

print(parameters)
cursor.close()
connection.close()

print(parameters)