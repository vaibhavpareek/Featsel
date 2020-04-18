#Set up the Environment
echo "alias FeatSel='python3 $PWD/main.py'" >> ~/.bashrc
#Set the PWD for the main file
from os import system,rename,popen
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