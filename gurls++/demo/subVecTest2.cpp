#include <iostream>
#include <string>

#include "gurls++/gvec.h"
#include "gurls++/gmat2d.h"

using namespace gurls;
using namespace std;
typedef double T;


int main()
{
    gMat2D<T> M(gMat2D<T>::zeros(10,10));
    gMat2D<T> P(gMat2D<T>::zeros(5,5));

    // Initialize P
    gVec<T> tmpCol(10);
    gVec<T> tmpCol1(5);
    for ( int i = 0 ; i < 5 ; ++i )
    {
        tmpCol = M(5 + i);
        tmpCol1 = tmpCol.subvec( (unsigned int) 5 ,  (unsigned int) 0 );
        P.setColumn(  tmpCol1 , (long unsigned int) i);
    }
    cout << "P initialized: " << P << endl;
}