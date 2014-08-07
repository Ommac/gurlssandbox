#include <iostream>
//#include <string>

#include "gurls++/blas_lapack.h"
//#include "quickanddirty.h"

//using namespace gurls;
//using namespace std;

//int main()
int main(int argc, char* argv[])
{
    //typedef double T;

    // Declarations
    std::cout << "0";

    double* x = new double;
    double* y = new double;
    double* c = new double;
    double* s = new double;
   
    std::cout << "1";
    // x coordinates
    *x = 1;
    
    std::cout << "2";
    // y coordinates
    *y = 1;
    
    std::cout << "3";
    //theta = 0
    *c = 0;
    *s = 1;
       
    std::cout << "4";
    // Display
    std::cout << "Initial conditions" << std::endl;        
    std::cout << "x = " << *x << std::endl;
    std::cout << "y = " << *y << std::endl;        
    std::cout << "cos(theta) = " << *c << std::endl;
    std::cout << "sin(theta) = " << *s << std::endl;
    
    int one = 1;
    
    // Apply rotations
    drot_(&one, x, &one, y, &one, c, s);
    
    // Rotated
    std::cout << "Final conditions" << std::endl;        
    std::cout << "x' = " << *x << std::endl;
    std::cout << "y' = " << *y << std::endl;
            
    return 0;
}
