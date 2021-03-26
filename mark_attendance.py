from datetime import datetime
from connect import connect_mysql


def mark_attendance(label, id):

    time = datetime.now().strftime("%H:%M:%S")
    date = datetime.today().strftime("%d-%m-%y")


    print(date, "Sdfsdfsd")

    sql = "INSERT INTO students (id, name, timeofatnd, dateofatnd) VALUES (%s, %s, %s, %s)"
    val = (id, label, time, date)

    conn, cursor = connect_mysql("localhost", "root", "root", "pragyanand")
    cursor.execute(sql, val)


    conn.commit()
    cursor.execute("select * from students")


    # res = cursor.fetchall()
    # for i in res:
    #     print(i)

    cursor.close()

    conn.rollback()


if __name__ == '__main__':
    print("")