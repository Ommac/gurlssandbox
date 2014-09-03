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

    cout << "M initialized: " <<endl<< M << endl;

    //gMat2D<T> N(gMat2D<T>::zeros(10,5));
    gMat2D<T> N(10,5);
    gMat2D<T> P(gMat2D<T>::zeros(10,5));

    // Initialize N
    M.submatrix(N , 10 , 5);
    cout << "N initialized: " <<endl<< N << endl;

    // Initialize P
    gVec<T> tmpCol(10);
    gVec<T> tmpCol1(5);
    for ( int i = 0 ; i < 5 ; ++i )
    {
        tmpCol = M(5 + i);
        
//     cout << tmpCol.subvec( (unsigned int) n_pretr ,  (unsigned int) 0);

        gVec<T> locs(10);
        for (int j = 0 ; j < 10 ; ++j)
            locs[j] = j;
        gVec<T>& tmpCol2 = tmpCol.copyLocations(locs);

//         tmpCol1 = tmpCol.subvec( (unsigned int) 50 ,  (unsigned int) 0 );
        //cout << "tmpCol1: " << tmpCol1 << endl;
//         P.setColumn(  tmpCol1 , (long unsigned int) i);
        P.setColumn(  tmpCol2 , (long unsigned int) i);
    }
    cout << "P initialized: " <<endl<< P << endl;    
}