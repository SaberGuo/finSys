import sqlite3

conn = sqlite3.connect('data/processed/zz500_data.db')
cursor = conn.cursor()

# Check daily_data
cursor.execute('SELECT * FROM daily_data LIMIT 3')
print('Daily data samples:')
for row in cursor.fetchall():
    print(row)

# Get columns
cursor.execute('PRAGMA table_info(daily_data)')
print('\nColumns in daily_data:')
for col in cursor.fetchall():
    print(f'  {col[1]} ({col[2]})')

# Get sample tickers
cursor.execute('SELECT DISTINCT code FROM daily_data LIMIT 10')
print('\nSample tickers:')
for row in cursor.fetchall():
    print(f'  {row[0]}')

# Count records
cursor.execute('SELECT COUNT(*) FROM daily_data')
print(f'\nTotal records: {cursor.fetchone()[0]}')

conn.close()
