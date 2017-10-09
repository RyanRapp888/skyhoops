
import sys
import matplotlib.pyplot as plt

def main():
   if(len(sys.argv) == 2):
      plt.plotfile(sys.argv[1],delimiter=',',cols=(0,1),names=('col1','col2'),marker='o')
      plt.show()
   
main()