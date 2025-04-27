from fabric import Connection, Config
import getpass
import time

#Prompt for the password
#password = getpass.getpass("Enter your root password: ")
password = "Arthrex1"

conn = Connection(host = "192.168.1.235", user="arthrex", connect_kwargs={'password': password})

for caseIndex in range(1, 10):
    time.sleep(5)
    source_path = "/home/arthrex/system_logs.txt"
    destination_path = f"case{caseIndex}.txt"
    conn.run(f'cp {source_path} /home/arthrex/cases/{destination_path}')