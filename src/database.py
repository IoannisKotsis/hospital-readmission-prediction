# Import necessary modules
from dotenv import load_dotenv
import os
import psycopg2

# Load environment variables from the .env file
load_dotenv()


def get_connection():

    # Reads from env variables
    db_host = os.getenv("DB_HOST")
    db_name = os.getenv("DB_NAME")
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_port = os.getenv("DB_PORT")

    try:
        # Creates the connection
        connection = psycopg2.connect(
            host = db_host,
            dbname = db_name,
            user = db_user,
            password = db_password,
            port = db_port
        )

        return connection
    except Exception as e:
        print(f"Error: {e}")
        raise

    
def insert_prediction(data):

    columns = ", ".join(data.keys())
    placeholders = ", ".join(['%s'] * len(data))
    values = tuple(data.values())

    # Connection and cursor creation
    connection = get_connection()
    cursor = connection.cursor()

    # Query creation
    query  = f"INSERT INTO readmission_info ({columns}) VALUES ({placeholders})"

    # Query execution
    try:
        cursor.execute(query, values)
        connection.commit()
    except Exception as e:
        print(f"Error: {e}")
        connection.rollback()
        raise
    finally:
        cursor.close()
        connection.close()



if __name__ == "__main__":
    connection = get_connection()
    print(connection)
    connection.close()