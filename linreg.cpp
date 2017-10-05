#include <iostream>
#include <vector>
#include <memory>
#include <fstream>
#include <sstream>
#include <string>
#include "../../../Eigen/Dense"

using namespace Eigen;

class LinearRegressionSolver
{
   public:
      void SetData( std::shared_ptr< std::vector<double> > x, 
                    std::shared_ptr< std::vector<double> > y)
      {
        m_x = x;
        m_y = y;
        ValidateSourceData();
      }
      
      void SolveUsingClosedFormNormalEquation( std::vector<double> &out_thetas)
      {
         // Right now, i'm handling the case of hypothesis functions in the form of 
         // theta0 + theta1*x 
         // x can have many features, in this case, I've fixed it to one
         // thetas.size() will be == to the num of features in x
         MatrixXd tmp_thetas(1,1);
          
      }
	  
      void PrintSourceData()
      {
         if(ValidateSourceData())
         {
            int idx(0);
            for(int aa = 0; aa < m_x->size(); aa++)
            {
               std::cout << "dat[" << aa << "] = " << (*m_x)[aa] << " , " << (*m_y)[aa] << std::endl;
            }
         } 
      }

   private:
      bool ValidateSourceData()
      {
         if(m_x->size() == 0){ std::cout << "Error: x array is empty\n"; return false;}

         if(m_y->size() == 0){ std::cout << "Error: y array is empty\n"; return false;}

         if(m_x->size() != m_y->size())
         {
            std::cout << "Error: x and y don't have the same number of data elements\n";
            return false;
         }
         return true;
      }
      std::shared_ptr<std::vector<double> > m_x;
      std::shared_ptr<std::vector<double> > m_y;
       
};

int main(int argc, char *argv[])
{
   if(argc != 2)
   {
      std::cout << "usage: linreg <csv file>" << std::endl;
      return EXIT_FAILURE;
   }
 
   std::ifstream input_file(argv[1]);
   if(!input_file.is_open())
   {
      std::cout << "Failed to open input file" << std::endl;
      return EXIT_FAILURE;
   } 

   auto input_xdata = std::make_shared<std::vector<double> >();
   auto input_ydata = std::make_shared<std::vector<double> >();

   std::string curline;
   while( getline(input_file, curline) )
   {
      if(curline.find(',') == std::string::npos) continue;

      int aa =0; 
      int curlen = curline.size();

      for(int aa=0; aa < curlen; aa++)
      {
         if(curline[aa] == ',') curline[aa] = ' '; 
      }

      std::istringstream istr(curline);
      double c1(-7),c2(-7);
      istr >> c1 >> c2;

      input_xdata->emplace_back(c1); 
      input_ydata->emplace_back(c2); 
   }

   LinearRegressionSolver lrs; 
   lrs.SetData(input_xdata,input_ydata);
   lrs.PrintSourceData();
 
}
