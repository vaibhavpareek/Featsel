pwd = '/root/Desktop/Vaibhav-tools/Featsel'
from pandas import read_csv,get_dummies,DataFrame,Series,concat
import cv2
import sys
from seaborn import heatmap,pairplot,set
from os import system,popen,path
from sklearn.feature_selection import mutual_info_regression as mi
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
from statsmodels.api import OLS
from pyfiglet import Figlet
from clint.textui import colored
import csv
set()

def banner():
	banner1 = Figlet(font='sblood')
	print(colored.red(banner1.renderText("FeatSel")))
	banner2 = Figlet(font='digital')
	print(colored.blue(banner2.renderText(" |||Feature Selector Learner|||")))
	print(colored.yellow(banner2.renderText(" 		By - Vaibhav Pareek")))

def exit_banner():
	banner1 = Figlet(font='digital')
	print("\n")
	print(colored.yellow(banner1.renderText("Thanks for Using FeatSel")))

def version():
	print("FeatSel 1.0.0")
	exit_banner()
	sys.exit(0)

def workspace():
	try:
		while True:
			system("clear")
			banner()
			ch = input("Workspace (New/Old) : ").lower()
			w_name = input("WorkSpace Name : ")
			if(ch=='o' or ch=='old'):
				f = open(str(pwd)+"/workspace.txt","r")
				lines = f.readlines()
				f.close()
				for w_name in lines:
					print("WorkSpace loaded Successfully")
				break		
			elif(ch=='n' or ch=='new'):
				f = open(str(pwd)+"/workspace.txt","w+")
				f.writelines(str(w_name))
				f.close()
				system("mkdir -p "+str(pwd)+"/Workspaces/"+str(w_name)+"/images/")
				print("Workspaces created Successfully")
				break
		return w_name
	except KeyboardInterrupt:
		exit_banner()
		sys.exit(0)
	except Exception as e:
		print(e)
		print("Error Creating Workspace")
		exit_banner()
		sys.exit(0)





def pre_process_dataset():
	try:
		wk = workspace()
		input("Press any Key to Continue..")
		system("clear")
		print("""
	DataPreprocessing Includes:
	Data Cleaning
	Imputation(Replace Missing Values)
	Dealing with Strings in the DataSet
	Categorical Varibales Encoding (ONE HOT ENCODING)
	Resolving Dependancy after or before dummy variable (Dummy Variable Trap)
		 """)
		dtset = input("Location of the Dataset : ")
		pth = str(pwd)+"/Demodataset/"+str(dtset)
		if(path.isfile(pth)==False):
			print("No Such File Exist in System")
			exit_banner()
			sys.exit(0)
		dtset = read_csv(pth)
		org_dataset=dtset
		print("Columns List ")
		k=1
		for i in dtset.columns:
			print(str(k)+" "+i)
			k=k+1
		y = input("Choose y (eg:" + dtset.columns[0]+ ") : ")
		y_set = dtset[y]
		k=1
		str_feat=[]
		for i in dtset.columns:
			if type(dtset[i][0])==str:
				str_feat.append(i)
				print(str(k)+" " + i)
				k=k+1
		ch = list(map(int, input("Select Unuseful Feature (eg : 1,2): ").split(',')))
		for i in ch:
			dtset.drop(str_feat[i-1],axis=1,inplace=True)
		print("Dropped Unuseful Data")
		k=1
		dtset.drop(y,axis=1,inplace=True)
		print("Categorical Variables")
		ct_var = []
		res = []
		for i in dtset.columns:
			if(len(Series(dtset[i]).unique())<=10):
				print(str(k)+" "+str(i))
				ct_var.append(i)
				k=k+1
			else:
				print(i)
				res.append(DataFrame(dtset[i]))
		ch = input("ONE HOT ENCODING (y/n) : ").lower()
		if(ch=='y' or ch =='yes'):
			for i in ct_var:
				tr = dtset[i]
				tr = get_dummies(tr,drop_first=True)
				res.append(DataFrame(tr))
		print("Done Data Cleaning")
		res = concat(res,axis=1)
		res.to_csv(str(pwd)+"/Workspaces/"+str(wk)+'/processed_data.csv')
		print("Data may not be free from missing values.")
	except KeyboardInterrupt :
		exit_banner()
		sys.exit(0)
	except Exception as e:
		print(e)
		print("Some Error Occured")
		exit_banner()
		sys.exit(0)




def filter(dataset,req,wk):
	try:
		count = 1
		print("Features Available")
		for i in dataset.columns:
			print(str(count) + " "+str(i))
			count = count+ 1
		while True:
			index = int(input("Mention Target Feature [Number]: "))
			if index<1 or index>len(dataset.columns):
				print("Index should be among the list only")
			else:
				break
		re = input("Display Correlation Between Features (y/n) : ").lower()
		if(re=="y" or re=="yes"):
			print(dataset.corr())
		dataset.corr().to_csv(str(pwd)+"/Workspaces/"+str(wk)+"/Correlation.csv",index=True)
		graph = heatmap(dataset.corr(),annot=True).get_figure()
		graph.savefig(str(pwd)+"/Workspaces/"+str(wk)+'/images/Correlation.png')
		X = get_dummies(dataset,drop_first=True)
		y = dataset[str(dataset.columns[index-1])]
		mel = mi(X,y)
		mel = Series(mel)
		plt.figure()
		graph1 = mel.plot.bar(figsize=(10,4)).get_figure()
		graph1.savefig(str(pwd)+"/Workspaces/"+str(wk)+'/images/mutualreg.png')
		plt.figure()
		graph2 = pairplot(dataset).savefig(str(pwd)+"/Workspaces/"+str(wk)+'/images/pairs.png')
		print("Visualize Graphs in "+str(wk)+" folder")
	except Exception as e:
		print(e)
		print("Some Error Occured : Filter Method")

def filter_help():
	print("\033[1;33;48m")
	print("""
	Filter Method is used for the feature Selection
	NOte: 
	1. Correlation -> which tells how the labels are related to each other.Require to find the relation.
	2. mutual_info_regression-> It will help to determine how the independant feature will be able to determine the target feature.
	3. Constant -> it gives the confirmation that constant variable will not decide the target feature.
	4. Quasi Constant -> Removing the threshold value , we can use it to determine whether X is suffucient to determine the target result or not.
	""")


def wrapper(dataset,req,wk):
	try:
		if(req==False):
			count = 1
			print("Features Available")
			for i in dataset.columns:
				print(str(count) + " "+str(i))
				count = count+ 1
			while True:
				index = int(input("Mention Target Feature [Number]: "))
				if index<1 or index>len(dataset.columns):
					print("Index should be among the list only")
				else:
					break
			X = get_dummies(dataset,drop_first=True)
			y = dataset[str(dataset.columns[index-1])]
			model = OLS(endog=y,exog=X).fit()
			f = open(str(pwd)+"/Workspaces/"+str(wk)+"/summaryOLS.txt","w+")
			f.write(str(model.summary()))
			f.close()
			print(model.summary())
			from numpy import array
			dt = model.pvalues.to_csv(str(pwd)+"/Workspaces/"+str(wk)+'/impfeatures.csv',header=True)
			dt = read_csv(str(pwd)+"/Workspaces/"+str(wk)+'/impfeatures.csv')
			dt = array(dt)
			print("Important Feature for Predicting Target Feature\n")
			f = open(str(pwd)+"/Workspaces/"+str(wk)+"/impfeatures.txt","w+")
			f.write("Important Features")
			for i in dt:
				if i[1]<0.05:
					print(str(i[0]))
					f.write(str(str(i[0])+" " + str(i[1]))+"\n")
			f.close()
		else:
			print("Trained Model")
	except Exception as e:
		print(e)
		print("Error Occured in Wrapper")

def wrapper_help():
	print("\033[1;35;48m")
	print("""
		Wrapper Method is used for the feature selection
		Note:
		*Wrapper method is mainly used when we are concern more about the performance ,then we use this method to do dimensionality reduction.
		*This method is performs elemination of features which are not impacting the result though if they get removed.
		*This method uses the OLS (Ordinary Least Square) method performs this task.
		*LinearRegression is commonly known as OLS.
		*OLS will work on the concepts of P-value and Adjusted Value which should be less than the suffucient value(Generally 5%). 
		*Wrapper Method use with the aim to provide only th features which are solely capable of determining the target feature.
		*OLS is the algorithm for using Wrapper Method. 
		""")

def embed(dataset,req,wk):
	try:
		if(req==True):
			print("Model Trained")
		elif(req==False):
			count = 1
			print("Features Available")
			for i in dataset.columns:
				print(str(count) + " "+str(i))
				count = count+ 1
			while True:
				index = int(input("Mention Target Feature [Number]: "))
				if index<1 or index>len(dataset.columns):
					print("Note: Index should be among the list only")
				else:
					break	
			if(len(dataset.columns)==2):
				y = dataset[dataset.columns[index-1]] 
				if(index==1):
					X = dataset[dataset.columns[index]]
				else:
					X = dataset[dataset.columns[0]]
			else:
				X = get_dummies(dataset.loc[:,dataset.columns!=dataset.columns[index-1]],drop_first=True)
				y = dataset[dataset.columns[index-1]]
			print("""
				Methods or ways of Embedded Method
				Press 1. Coeffecient or Weight
				Press 2. Lasso (L1) Regulation
				Press 3. To Exit
				""")
			while True:
				ch = int(input("Choice : "))
				if(ch==3):
					exit_banner()
					sys.exit(0)
				elif(ch>3 or ch<1):
					print("Choose among the available options only")
				else:
					break
			while True:
				test_s = int(input("Percentage of Records for Testing Case(1%-100%) : "))
				if(test_s<1 or test_s>100):
					print("Range is Incorrect\n")
				else:
					test_s = float(test_s/100)
					X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_s,random_state=42)
					if(ch==1):
						model = LinearRegression()
						model.fit(X_train,y_train)
						y_pred = model.predict(X_test)
						plt.scatter(y_pred,y_test,color='red')
						plt.plot(y_pred,y_test)
						plt.title('DataSet Coeffecient')
						plt.savefig(str(pwd)+"/Workspaces/"+str(wk)+'/images/linearmodel.png')
						f = open(str(pwd)+"/Workspaces/"+str(wk)+'/coeff.txt',"w+")
						cols = list(X.columns)
						print("Features  :   Coeffecient")
						f.write("Features  :   Coeffecient\n")
						for (i,j) in zip(list(model.coef_),cols):
							print(str(j)+" : "+str(i))
							f.write(str(j)+" : "+str(i))
						f.write("\nIntercept : "+str(model.intercept_)+"\n")
						print("\nIntercept : ",model.intercept_)
						print("\nImportant Features based on their Coeffecients/Weights\n")
						f.write("\nImportant Features based on their Coeffecients/Weights\n")
						for (i,j) in zip(list(model.coef_),cols):
							if(i>0):
								print(str(j) + " : "+ str(i))
								f.write(str(j) + " : "+ str(i))
						if(len(model.coef_)>1):
							print("\nCDF Display\n")
							f.write("\nCDF Display\n")
							cdf = DataFrame(model.coef_,dataset.columns,columns=['Coeffecients'])
							f.write(str(cdf))
							print(cdf)
						f.close()
					elif(ch==2):
						sel = SelectFromModel(Lasso(alpha=100))
						sel.fit(X_train,y_train)
						f = open(str(pwd)+"/Workspaces/"+str(wk)+'/lasso.txt',"w+")
						print("Lasso Model Evaluation on Feature Selection\n")
						f.write("Lasso Model Evaluation on Feature Selection\n")
						for (i,j) in zip(list(sel.get_support()),list(X.columns)):
							if(i>0):
								print(str(j) + " : "+ str(i))
								f.write(str(j) + " : "+ str(i))
						print("Estimator_Coeff : ",sel.estimator_.coef_)
						f.write("Estimator_Coeff : ",sel.estimator_.coef_)	
						f.close()
					break		
	except KeyboardInterrupt:
		exit_banner()
		sys.exit(0)
	except Exception as e:
		print(e)
		print("Some Error Occured in Embedded")

def embed_help():
	print("\033[1;32;48m")
	print("""
	Embedded Method is used for the feature Selection
	NOte: 
	1. Coeffecient or Weight -> which helps in determining the feature weight(power) to decide the target feature.
	2. Lasso(R1 Regulation) -> It use the Lasso linear model to train the system to decide the feature which could be use to get desired result. 
	Embedded method internally always train the model before predicting the actual independant features to determine the target feature.
	This method is slow in progression because it takes time to train the model.
	""")
def tui_interface(tui):
	begin = True;
	while(tui):
		system("clear")
		banner()
		try:
			print("\033[1;34;48m")
			if(begin==True):
				wk = workspace()
				begin=False
			print("Feature Selection Methods")
			print("""
				Hint: Use FeatSel --list to view all locally available datasets
				Use : FeatSel --pre or FeatSel -p (Data PreProcessing)
				""")
			print("1. Filter Method")
			print("2. Wrapper Method")
			print("3. Embedded Method")
			print("4. Exit TUI Mode")
			ch = int(input("Enter Choice : "))
			if(ch==4):
				exit_banner()
				sys.exit(0)
			elif(ch>=1 and ch<4):
				choice = input("Local Dataset or Remote Dataset (l/r) : ").lower()
				if(choice=='l' or choice=='local'):
					data = input("Name of the DataSet : ")
					if(isavail(str(data))):
						dataset = read_csv(str(pwd)+"/Demodataset/"+str(data))
						if(ch==1):
							filter(dataset,False,wk)
							input("Press Any Key to Continue ... ")
						elif(ch==2):
							wrapper(dataset,False,wk)
							input("Press Any Key to Continue ... ")
						elif(ch==3):
							embed(dataset,False,wk)
							input("Press Any Key to Continue ... ")
					else:
						print("Dataset is not present")
				elif(ch=='r' or ch=='remote'):
					dataset = read_csv(input("Location/Path of the DataSet : "))
					if(ch==1):
							filter(dataset,False,wk)
					elif(ch==2):
						wrapper(dataset,False,wk)
					elif(ch==3):
						embed(dataset,False,wk)
				else:
					print("No Such Option available")				
		except KeyboardInterrupt:
			exit_banner()
			sys.exit(0)	
		except FileNotFoundError:
			print("DataSet Not Found")

def help():
	try:	
		system("chmod +x "+str(pwd)+"/manual")
		system("man "+str(pwd)+"/./manual")
		exit_banner(0)
	except:
		sys.exit(0)

def setup():
	try:
		system("python3 setup.py")
		print("Environment Setup Successfully")
		exit_banner()
		sys.exit(0)
	except FileNotFoundError:
		print("Setup file is missing")
		sys.exit(0)

def list_dataset():
	try:
		print("List of Dataset's Locally Present")
		print("\033[1;33;48m")
		pwd_list = str(pwd)+"/Demodataset"
		ls = popen('ls '+str(pwd_list)).read()
		if(len(ls)==0):
			print("--no-local dataset")
		else:
			print(ls)
		exit_banner()
		sys.exit(0)
	except Exception as e:
		print(e)
		print("Some Error Occured in Listing Dataset")
		sys.exit()	

def isavail(dataset):
	system('ls '+str(pwd)+'/Demodataset/ > temp.txt')
	dataset=dataset+"\n"
	f = open("temp.txt","r")
	Lines = f.readlines()
	flag = True
	if(dataset in Lines):
		flag=False
	f.close()
	system("rm temp.txt")
	if(flag):
		print("No Dataset with such Name is present in Local center")
		return False
	else:
		return True 


def delete_dataset():
	try:
		print("List of Dataset's Locally Present")
		print("\033[1;33;48m")
		pwd_list = str(pwd)+"/Demodataset"
		ls = popen('ls '+str(pwd_list)).read()
		if(len(ls)==0):
			print("--no-local dataset")
		else:
			print(ls)
		ty = input("DataSet Name wants to delete : ")
		system('ls '+str(pwd)+'/Demodataset/ > temp.txt')
		f = open("temp.txt","r")
		flag = True
		for i in f:
			if str(ty) in i:
				system("rm "+str(pwd)+"/Demodataset/"+str(ty))
				print("Dataset Removed Successfully")
				flag = False
				break
		if(flag):
			print("No Dataset with such Name is present in Local center")
		f.close()
		system("rm temp.txt")
		exit_banner()
		sys.exit(0)
	except Exception as e:
		print(e)
		print("Some Error Occured in Deleting Dataset")
		sys.exit()	



def add_dataset():
	print("\033[1;32;48m")
	print("Add DataSet to Local DataBase")
	df = input("Location of the DataSet : ")
	df_name = df.split("/")
	try:
		system('ls '+str(pwd)+'/Demodataset/ > temp.txt')
		f = open("temp.txt","r")
		flag = True
		for i in f:
			if str(df_name[-1]) in i:
				print("DataSet with this name already exist.")
				flag = False
		if(flag):
			system("cp "+df+" "+str(pwd)+"/Demodataset/")
			print("Successfully Added")
		f.close()	
		system('rm temp.txt')
		exit_banner()
		sys.exit(0)
	except Exception as e:
		print("Error Occured - Dataset Not Added")
		sys.exit()	

def main():
	arg=list(sys.argv)
	if(len(arg) > 2):
		print("More Arguments are passed then Required")
		exit_banner()
		sys.exit(0)
	elif(len(arg)<2):
		print("FeatSel works with options only , Refer FeatSel --help to know more.")
	else:
		try:				
			if(arg[1]=="--help" or arg[1]=="-h"):
				help()
			elif(arg[1]=="--tui" or arg[1]=="-t"):
				tui_interface(True)
			elif(arg[1]=="--banner" or arg[1]=="-b"):
				banner()
			elif(arg[1]=="--install" or arg[1]=="-i"):
				try:
					system("pip3 install -r requirement.txt")
					print("All the Modules installed Successfully")
				except:
					print("Try Again : Issue Occured installing modules")
					sys.exit(0)
			elif(arg[1]=="--filter"or arg[1]=="-f"):
				print("\033[1;34;48m")
				wk = workspace()
				print("Filter Method for Feature Selection")
				print("Hint: Use FeatSel --list to view all locally available datasets")
				ch = input("Local Dataset or Remote Dataset (l/r) : ").lower()
				if(ch=='l' or ch=='local'):
					data = input("Name of the DataSet : ")
					if(isavail(str(data))):
						dataset = read_csv(str(pwd)+"/Demodataset/"+str(data))
						filter(dataset,False,wk)
					else:
						print("Dataset is not present")
				elif(ch=='r' or ch=='remote'):
					dataset = read_csv(input("Location/Path of the DataSet : "))
					filter(dataset,wk)
				else:
					print("No Such Option available")
				val = input("Learn - Filter Method (y/n) : ").lower()
				if (val=='y' or val=='yes'):
					filter_help()
				exit_banner()
				sys.exit(0)
			elif(arg[1]=="--wrapper"or arg[1]=="-w"):
				print("\033[1;34;48m")
				wk=workspace()
				print("Wrapper Method for Feature Selection")
				print("Hint: Use FeatSel --list to view all locally available datasets")
				tr = input("Do you have trained Model (y/n) : ").lower()
				if(tr=='y' or tr=='yes'):
					loc = input("Location/Path of Trained Model : ")
					wrapper(loc,True,wk)
				elif(tr =='n' or tr=='no'):
					ch = input("Local Dataset or Remote Dataset (l/r) : ").lower()
					if(ch=='l' or ch=='local'):
						data = input("Name of the DataSet : ")
						if(isavail(str(data))):
							dataset = read_csv(str(pwd)+"/Demodataset/"+str(data))
							wrapper(dataset,False,wk)
						else:
							print("Dataset is not present")
					elif(ch=='r' or ch=='remote'):
						dataset = read_csv(input("Location/Path of the DataSet : "))
						wrapper(dataset,False,wk)
					else:
						print("No Such Option available")
				else:
					print("No Such Option available")
				val = input("Learn - Wrapper Method (y/n) : ").lower()
				if (val=='y' or val=='yes'):
					wrapper_help()
				exit_banner()
				sys.exit(0)
			elif(arg[1]=="--embed"or arg[1]=="-e"):
				print("\033[1;34;48m")
				wk=workspace()
				print("Embedded Method for Feature Selection")
				print("Hint: Use FeatSel --list to view all locally available datasets")
				tr = input("Do you have trained Model (y/n) : ").lower()
				if(tr=='y' or tr=='yes'):
					loc = input("Location/Path of Trained Model : ")
					embed(loc,True,wk)
				elif(tr =='n' or tr=='no'):
					ch = input("Local Dataset or Remote Dataset (l/r) : ").lower()
					if(ch=='l' or ch=='local'):
						data = input("Name of the DataSet : ")
						if(isavail(str(data))):
							dataset = read_csv(str(pwd)+"/Demodataset/"+str(data))
							embed(dataset,False,wk)
						else:
							print("Dataset is not present")
					elif(ch=='r' or ch=='remote'):
						dataset = read_csv(input("Location/Path of the DataSet : "))
						embed(dataset,False,wk)
					else:
						print("No Such Option available")
				else:
					print("No Such Option available")
				val = input("Learn - Embedded Method (y/n) : ").lower()
				if (val=='y' or val=='yes'):
					embed_help()
				exit_banner()
				sys.exit(0)
			elif(arg[1]=="--version"or arg[1]=="-v"):
				print("\033[1;34;48m")
				version()
			elif(arg[1]=="--setup"or arg[1]=="-s"):
				print("\033[1;34;48m")
				setup()
			elif(arg[1]=="--delete"or arg[1]=="-d"):
				print("\033[1;34;48m")
				delete_dataset()
			elif(arg[1]=="--list" or arg[1]=="-l"):
				print("\033[1;34;48m")
				list_dataset()
			elif(arg[1]=="--add" or arg[1]=="-a"):
				print("\033[1;34;48m")
				add_dataset()
			elif(arg[1]=='--pre' or arg[1]=='-p'):
				print("\033[1;34;48m")
				pre_process_dataset()
			else:
				print(str(arg[1]+" not supported by FeatSel"))
				exit_banner()
				sys.exit()
		except FileNotFoundError:
			print("DataSet Not Found")
			exit_banner()
			sys.exit(0)
		except Exception as e:
			print("Error."+str(e))
			print("Some Error Occured")
			exit_banner()
			sys.exit(0)
main()