#Set up the Environment
from os import system,rename,popen
comm = """
echo "alias FeatSel='python3 $PWD/main.py'" >> ~/.bashrc
"""
system(comm)
#Set the PWD for the main file
r = open("main.py","r")
w = open("temp.py","w")
k = 0
for i in r:
	if(k==0):
	    p = popen("pwd").read()
	    p = "pwd = '"+str(p[:len(p)-1])+"'\n"
	    w.write(str(p))
	else:
		w.write(i)
	k = k+1
system("rm main.py")
rename("temp.py","main.py")
w.close()
r.close()