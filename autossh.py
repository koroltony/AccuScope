from fabric import Connection, Config
import getpass
import time
import sys

'''

#Prompt for the password
#password = getpass.getpass("Enter your root password: ")
password = "Arthrex1"

# to make more robust, prompt for ip, user, password
conn = Connection(host = "192.168.1.235", user="arthrex", connect_kwargs={'password': password})

print("running")

conn.run('nohup python3 jtopTestScript.py > log.txt 2>&1 &', pty=False)

time.sleep(1)

print('running in background')

time.sleep(1)

conn.run("pkill -f jtopTestScript.py")

print('command killed')

# Download the system_logs.txt file
remote_path = './system_logs.txt'  # path on the system
local_path = './arthrex_system_logs.txt' # path in this folder
conn.get(remote_path, local_path)

print(f'Downloaded {remote_path} to {local_path}')
'''

if len(sys.argv) != 4:
    print("Usage: python autossh.py <ip> <username> <password>")
    sys.exit(1)

ip = sys.argv[1]
username = sys.argv[2]
password = sys.argv[3]

conn = Connection(
    host=ip,
    user=username,
    connect_kwargs={'password': password}
)

print("running")

conn.run('nohup python3 jtopTestScript.py > log.txt 2>&1 &', pty=False)

time.sleep(1)
print('running in background')

time.sleep(1)
conn.run("pkill -f jtopTestScript.py")
print('command killed')

remote_path = './system_logs.txt'
local_path = './arthrex_system_logs.txt'
conn.get(remote_path, local_path)

print(f'Downloaded {remote_path} to {local_path}')
