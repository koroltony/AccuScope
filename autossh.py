from fabric import Connection, Config
import getpass
import time

#Prompt for the password
#password = getpass.getpass("Enter your root password: ")
password = "Arthrex1"

conn = Connection(host = "192.168.1.235", user="arthrex", connect_kwargs={'password': password})

print("running")

conn.run('nohup python3 jtopTestScript.py > log.txt 2>&1 &', pty=False)

time.sleep(5)

conn.run("pkill -f jtopTestScript.py")