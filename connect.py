import mysql.connector


def connect_mysql(host, user, password, database):

  try:

    mydb = mysql.connector.connect(
      host= host,
      user= user,
      password= password,
      database = database
    )

    mycursor = mydb.cursor(buffered=True)
    return mydb, mycursor

  except Exception as e:
    print(e)
