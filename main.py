from pandas import read_csv,get_dummies,DataFrame,Series
import cv2
import sys
from seaborn import heatmap,pairplot
from os import system,popen
from sklearn.feature_selection import mutual_info_regression as mi
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
from statsmodels.api import OLS
from pyfiglet import Figlet
from clint.textui import colored

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

def filter(dataset):
	print(dataset.corr())
	graph = heatmap(dataset.corr(),annot=True).get_figure()
	graph.savefig('SalaryData'+'.png')
	system('eog SalaryData.png &')
	X = get_dummies(dataset,drop_first=True)
	y = dataset['Salary']
	mel = mi(X,y)
	mel = Series(mel)
	plt.figure()
	# mel.index = 'YearsExperience'
	graph1 = mel.plot.bar(figsize=(10,4)).get_figure()
	graph1.savefig('mutualreg'+'.png')
	system('eog mutualreg.png &')
	plt.figure()
	pairplot(dataset)
	graph2 = mel.plot.bar(figsize=(10,4)).get_figure()
	graph2.savefig('pairs'+'.png')
	system('eog pairs.png &')

def filter_help():
	print("\033[1;33;48m")
	print("""
	Filter Method is used for the feature Selection
	NOte: 
	1. Correlation -> which tells how the labels are related to each other.Require to find the relation.
	2. mutual_info_regression-> It will help to determine how the independant feature will be able to determine the target feature.
	3. Constant -> it gives the confirmation that constant variable will not decide the target feature.
	4. Quasi Constant -> Removing the threshold value , we can use it to determine whether X is suffucient to determine the target result.
	""")


def wrapper(dataset):
	X = get_dummies(dataset,drop_first=True)
	y = dataset['Salary']
	model = OLS(endog=y,exog=X).fit()
	print(model.summary())


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

def embed(dataset):
	X = get_dummies(dataset,drop_first=True)
	y = dataset['Salary']
	print("""
		Press 1. Coeffecient or Weight
		Press 2. Lasso (L1) Regulation
		""")
	ch = int(input("Choice : "))
	X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,random_state=42)
	if(ch==1):
		model = LinearRegression()
		model.fit(X_train.reshape(21,1),y_train)
		y_pred = model.predict(X_test.reshape(9,1))
		plt.scatter(X_test,y_test,color='red')
		plt.plot(X_test,y_pred)
		plt.title('Salary VS Experience Prediction')
		plt.savefig('model.png')
		for i in model.coef_:
			print("Coeffecient : ",i)
		print("Intercept : ",model.intercept_)
		cdf = DataFrame(model.coef_,dataset[['YearsExperience']].columns,columns=['Coeffecients'])
		print(cdf)
	elif(ch==2):
		sel = SelectFromModel(Lasso(alpha=100))
		sel.fit(X_train,y_train)
		print(sel.get_support())
		print(sel.estimator_.coef_)
		
	else:
		print("Che among the available options only")


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
	while (tui):
		banner()
		try:
			print("\033[1;34;48m")
			dataset = read_csv(input("Path to the DataSet : "))
			print("Feature Selection Methods")
			print("1. Filter Method")
			print("2. Wrapper Method")
			print("3. Embedded Method")
			print("4. Exit TUI Mode")
			ch = int(input("Enter Choice : "))
			if(ch==1):
				filter(dataset)
				val = lower(input("Learn - Filter Method (y/n) : "))
				if (val=='y' or val=='yes'):
					filter_help()
			elif(ch==2):
				wrapper(dataset)
				val = lower(input("Learn - Wrapper Method (y/n) : "))
				if (val=='y' or val=='yes'):
					wrapper_help()
			elif(ch==3):
				embed(dataset)
				val = lower(input("Learn - Embedded Method (y/n) : "))
				if (val=='y' or val=='yes'):
					embed_help()
			elif(ch==4):
				exit_banner()
				sys.exit(0)		
		except KeyboardInterrupt:
			exit_banner()
			sys.exit(0)	
		except FileNotFoundError:
			print("DataSet Not Found")

def help():
	try:	
		system("chmod +x manual")
		system("man ./manual")
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
		ls = popen('ls Demodataset/').read()
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

def add_dataset():
	print("\033[1;32;48m")
	print("Add DataSet to Local DataBase")
	df = input("Location of the DataSet : ")
	df_name = df.split("/")
	try:
		system('ls Demodataset/ > temp.txt')
		f = open("temp.txt","r")
		if(str(df_name[-1]) in f):
			print("DataSet with this name already exist.")
		else:
			system("cp "+df+" Demodataset/")
			print("Successfully Added")
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
				dataset = read_csv(input("Path to the DataSet : "))
				filter(dataset)
				val = lower(input("Learn - Filter Method (y/n) : "))
				if (val=='y' or val=='yes'):
					filter_help()
			elif(arg[1]=="--wrapper"or arg[1]=="-w"):
				print("\033[1;34;48m")
				dataset = read_csv(input("Path to the DataSet : "))
				wrapper(dataset)
				val = lower(input("Learn - Wrapper Method (y/n) : "))
				if (val=='y' or val=='yes'):
					wrapper_help()
			elif(arg[1]=="--embed"or arg[1]=="-e"):
				print("\033[1;34;48m")
				dataset = read_csv(input("Path to the DataSet : "))
				embed(dataset)
				val = lower(input("Learn - Embedded Method (y/n) : "))
				if (val=='y' or val=='yes'):
					filter_help()
			elif(arg[1]=="--version"or arg[1]=="-v"):
				print("\033[1;34;48m")
				version()
			elif(arg[1]=="--setup"or arg[1]=="-s"):
				print("\033[1;34;48m")
				setup()
			elif(arg[1]=="--list" or arg[1]=="-l"):
				print("\033[1;34;48m")
				list_dataset()
			elif(arg[1]=="--add" or arg[1]=="-a"):
				print("\033[1;34;48m")
				add_dataset()
			else:
				print(str(arg[1]+" not supported by FeatSel"))
				exit_banner()
				sys.exit()
		except FileNotFoundError:
			print("DataSet Not Found")
			exit_banner()
			sys.exit(0)
		except Exception as e:
			print("2."+str(e))
			print("Some Error Occured")
			exit_banner()
			sys.exit(0)
main()