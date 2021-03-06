 /*
  * The GURLS Package in C++
  *
  * Copyright (C) 2011-1013, IIT@MIT Lab
  * All rights reserved.
  *
  * authors:  Raffaello Camoriano
  * email:   raffaello.camoriano@iit.it
  * website: http://cbcl.mit.edu/IIT@MIT/IIT@MIT.html
  *
  * Redistribution and use in source and binary forms, with or without
  * modification, are permitted provided that the following conditions
  * are met:
  *
  *     * Redistributions of source code must retain the above
  *       copyright notice, this list of conditions and the following
  *       disclaimer.
  *     * Redistributions in binary form must reproduce the above
  *       copyright notice, this list of conditions and the following
  *       disclaimer in the documentation and/or other materials
  *       provided with the distribution.
  *     * Neither the name(s) of the copyright holders nor the names
  *       of its contributors or of the Massacusetts Institute of
  *       Technology or of the Italian Institute of Technology may be
  *       used to endorse or promote products derived from this software
  *       without specific prior written permission.
  *
  * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
  * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
  * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
  * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
  * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
  * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
  * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
  * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
  * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
  * POSSIBILITY OF SUCH DAMAGE.
  */


#ifndef _GURLS_RLSPRIMALRECINITCHOLESKYSCRATCH_H_
#define _GURLS_RLSPRIMALRECINITCHOLESKYSCRATCH_H_

#include "gurls++/optimization.h"

#include "gurls++/optmatrix.h"
#include "gurls++/optfunction.h"

#include "gurls++/utils.h"

namespace gurls
{

/**
 * \ingroup Optimization
 * \brief RLSPrimalRecInitCholeskyScratch is the sub-class of Optimizer that implements RLS with the primal formulation and computes the upper Cholesky factor of the regularized covariance matrix, starting from an empty XtX.
 */
template <typename T>
class RLSPrimalRecInitCholeskyScratch: public Optimizer<T>
{

public:
    /**
     * Computes a classifier for the primal formulation of RLS, computes and stores the Cholesky decomposition of the 
     * covariance matrix, starting from an empty XtX.
     * The regularization parameter is set to the one found in the field paramsel of opt.
	 * The variables necessary for further recursive update (R, W, b) are stored in the output structure
     * In case of multiclass problems, the regularizers need to be combined with the function specified inthe field singlelambda of opt
     *
     * \param X input data matrix
     * \param Y labels matrix
     * \param opt options with the following:
     *  - singlelambda (default)
     *  - paramsel (settable with the class ParamSelection and its subclasses)
     *
     * \return adds to opt the field optimizer which is a list containing the following fields:
     *  - W = matrix of coefficient vectors of rls estimator for each class
     *  - R = Upper Cholesky factor of the regularized covariance matrix (n*lambda*I)
     *  - b = Xty vector
     */
    GurlsOptionsList* execute(const gMat2D<T>& X, const gMat2D<T>& Y, const GurlsOptionsList& opt);
};


template <typename T>
GurlsOptionsList* RLSPrimalRecInitCholeskyScratch<T>::execute(const gMat2D<T>& X, const gMat2D<T>& Y, const GurlsOptionsList &opt)
{
    //	lambda = opt.singlelambda(opt.paramsel.lambdas);
    const gMat2D<T> &ll = opt.getOptValue<OptMatrix<gMat2D<T> > >("paramsel.lambdas");
    T lambda = opt.getOptAs<OptFunction>("singlelambda")->getValue(ll.getData(), ll.getSize());
    
    // NOTE: An error should be thrown if paramsel.lambdas is not defined

    //	[n,d] = size(X);
    const unsigned long n = opt.hasOpt("nTot")? static_cast<unsigned long>(opt.getOptAsNumber("nTot")) : X.rows();
    unsigned long d;
    unsigned long t;


    //	Retrieve d
    if(!opt.hasOpt("kernel.XtX"))
    {
        d = X.cols();
    }
    else
    {
        const gMat2D<T>& XtX_mat = opt.getOptValue<OptMatrix<gMat2D<T> > >("kernel.XtX");
        d = XtX_mat.cols();
    }

    //	Retrieve t
    if(!opt.hasOpt("kernel.Xty"))
    {
        t = Y.cols();
    }
    else
    {
        const gMat2D<T>& Xty_mat = opt.getOptValue<OptMatrix<gMat2D<T> > >("kernel.Xty");
        t = Xty_mat.cols();
    }

    // Compute R from scratch
    // R = chol((n*lambda)*eye(d));

    T coeff = n*lambda;
    
    // Define empty XtX vectorized matrix
    T* XtX;
    XtX = new T[d*d];    
    //const gMat2D<T>& XtX_mat(gMat2D<T>::zeros(d, d));
    copy(XtX, gMat2D<T>::zeros(d, d).getData(), d*d);
    
    axpy(d, (T)1.0, &coeff, XtX, C, d+1);	// Regularize XtX, adding n*lambda on the diagonal

    // Cholesky factorization

    // Initialize weights vector
    // W = zeros(d,1);    

    //	R = chol(XtX);
    T* R = new T[d*d];
    cholesky(XtX, d, d, R);	// Computes the upper Cholesky factor by default
    gMat2D<T> *R_mat = new gMat2D<T>(R, d, d, 0);
    
    delete[] XtX;
    delete[] R;

    // W = zeros(d,t);
    gMat2D<T> *W = new gMat2D<T>(gMat2D<T>::zeros(d, t));

    // b = zeros(d,t);
    gMat2D<T> *b = new gMat2D<T>(W);
    
    //Save results in OPT
    GurlsOptionsList* optimizer = new GurlsOptionsList("optimizer");

    // cfr.R = R;
    optimizer->addOpt("R", new OptMatrix<gMat2D<T> >(*R));
    
    //  cfr.W = W;
    optimizer->addOpt("W", new OptMatrix<gMat2D<T> >(*W));

    //	cfr.b = Xty;
    optimizer->addOpt("b", new OptMatrix<gMat2D<T> >(*Xty));

    return optimizer;
}


}
#endif // _GURLS_RLSPRIMALRECINITCHOLESKYSCRATCH_H_
