import sqlite3

conn = sqlite3.connect('data/processed/zz500_data.db')
cursor = conn.cursor()

# Get table names
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
print('Tables:', tables)

# Get sample data
if tables:
    table_name = tables[0][0]
    cursor.execute(f'SELECT * FROM {table_name} LIMIT 3')
    print(f'\nSample data from {table_name}:')
    for row in cursor.fetchall():
        print(row)

    # Get column names
    cursor.execute(f'PRAGMA table_info({table_name})')
    columns = cursor.fetchall()
    print(f'\nColumns in {table_name}:')
    for col in columns:
        print(f'  {col[1]} ({col[2]})')

conn.close()
