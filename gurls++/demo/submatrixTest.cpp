#include <iostream>
#include <string>

#include "gurls++/gvec.h"
#include "gurls++/gmat2d.h"

using namespace gurls;
using namespace std;
typedef double T;


int main()
{
    gMat2D<T> M(gMat2D<T>::zeros(10,10)+1);
    cout << "M initialized: " << endl << M << endl;

    gMat2D<T> N(gMat2D<T>::zeros(10,5));

    // Initialize N
    N.submatrix(M , 10 , 5);
    cout << "N initialized: " << endl << N << endl;
}