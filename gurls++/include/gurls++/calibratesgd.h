/*
 * The GURLS Package in C++
 *
 * Copyright (C) 2011-1013, IIT@MIT Lab
 * All rights reserved.
 *
 * authors:  M. Santoro
 * email:   msantoro@mit.edu
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


#ifndef _GURLS_CALIBRATESGD_H_
#define _GURLS_CALIBRATESGD_H_

#include "gurls++/options.h"
#include "gurls++/optlist.h"
#include "gurls++/gmat2d.h"
#include "gurls++/gmath.h"

#include "gurls++/paramsel.h"
#include "gurls++/perf.h"
#include "gurls++/gurls.h"

namespace gurls {

/**
 * \ingroup ParameterSelection
 * \brief ParamselCalibrateSGD is the sub-class of ParamSelection that implements parameter selection for pegasos
 */

template <typename T>
class ParamSelCalibrateSGD: public ParamSelection<T>{

public:
    /**
     * Performs parameter selection when one wants to solve the problem using rls_pegasos.
     * \param X input data matrix
     * \param Y labels matrix
     * \param opt options with the following:
     *  - subsize (default)
     *  - calibfile (default)
     *  - hoperf (default)
     *  - singlelambda (default)
     *  - nlambda (default)
     *
     * \return paramsel, a GurlsOptionList with the following fields:
     *  - lambdas = array of values of the regularization parameter lambda minimizing the validation error for each class
     *  - W = RLS coefficient vector
     */
    GurlsOptionsList* execute(const gMat2D<T>& X, const gMat2D<T>& Y, const GurlsOptionsList& opt);
};

template <typename T>
GurlsOptionsList *ParamSelCalibrateSGD<T>::execute(const gMat2D<T>& X, const gMat2D<T>& Y, const GurlsOptionsList &opt)
{
//    n_estimates = 1;
    const unsigned long n_estimates = 1;

//    [n,d] = size(X);
    const unsigned long n = X.rows();
    const unsigned long t = X.cols();

    GurlsOptionsList* tmp = new GurlsOptionsList("ParamSelCalibrateSGD", true);

    OptTaskSequence *seq = new OptTaskSequence();
    GurlsOptionsList * process = new GurlsOptionsList("processes", false);
    OptProcess* process1 = new OptProcess();

    *seq << "split:ho" << "kernel:linear" << "paramsel:hodual" << "optimizer:rlsdual";
    *process1 << GURLS::compute << GURLS::compute << GURLS::computeNsave << GURLS::computeNsave;

    tmp->addOpt("seq", seq);

    process->addOpt("one", process1);
    tmp->addOpt("processes", process);


    if(tmp->hasOpt("hoperf"))
        tmp->removeOpt("hoperf");
    if(tmp->hasOpt("singlelambda"))
        tmp->removeOpt("singlelambda");

    tmp->addOpt("hoperf", opt.getOptAsString("hoperf"));
    tmp->addOpt("singlelambda", new OptFunction(OptFunction::dynacast(opt.getOpt("singlelambda"))->getName()));


    GURLS g;

//        sub_size = opt.subsize;
    const int subsize = static_cast<int>(opt.getOptAsNumber("subsize"));

    unsigned long* idx = new unsigned long[n]; //will use only the first subsize elements
    T* lambdas = new T[n_estimates];

    gMat2D<T> Mx(subsize, t);
    gMat2D<T> My(subsize, Y.cols());

    //    for i = 1:n_estimates,
    for(unsigned long i=0; i<n_estimates; ++i)
    {
//        idx = randsample(n, sub_size);
        randperm(n, idx);

//        M = X(idx,:);
        subMatrixFromRows(X.getData(), n, t, idx, subsize, Mx.getData());

//        if ~exist([opt.calibfile '.mat'],'file')
//            fprintf('\n\tCalibrating...');
//            %% Step 1 : Hold out parameter selection in the dual
//            name = opt.calibfile;
//            tmp.hoperf = opt.hoperf;
//            tmp = defopt(name);
//            tmp.seq = {'split:ho','kernel:linear','paramsel:hodual','rls:dual'};
//            tmp.process{1} = [2,2,2,2];
//            tmp.singlelambda = opt.singlelambda;

//            gurls(M,y(idx,:),tmp,1);
        subMatrixFromRows(Y.getData(), Y.rows(), Y.cols(), idx, subsize, My.getData());

        g.run(Mx, My, *tmp, "one");

//        end
//        fprintf('\n\tLoading existing calibration');
//        load([opt.calibfile '.mat']);
//        lambdas(i) = opt.singlelambda(opt.paramsel.lambdas);

        const gMat2D<T> &ll = tmp->getOptValue<OptMatrix<gMat2D<T> > >("paramsel.lambdas");
        lambdas[i] = opt.getOptAs<OptFunction>("singlelambda")->getValue(ll.getData(), ll.getSize());

//        % Add rescaling
//    end
    }

    delete[] idx;

    GurlsOptionsList* paramsel;

    if(opt.hasOpt("paramsel"))
    {
        GurlsOptionsList* tmp_opt = new GurlsOptionsList("tmp");
        tmp_opt->copyOpt("paramsel", opt);

        paramsel = GurlsOptionsList::dynacast(tmp_opt->getOpt("paramsel"));
        tmp_opt->removeOpt("paramsel", false);
        delete tmp_opt;

        paramsel->removeOpt("lambdas");
        paramsel->removeOpt("W");
    }
    else
        paramsel = new GurlsOptionsList("paramsel");


//    params.lambdas = mean(lambdas);
    gMat2D<T> *lambda = new gMat2D<T>(1,1);
    lambda->getData()[0] = sumv(lambdas, n_estimates)/n_estimates;
    paramsel->addOpt("lambdas", new OptMatrix<gMat2D<T> >(*lambda));


//    params.W = opt.rls.W;
    GurlsOptionsList* rls = tmp->getOptAs<GurlsOptionsList>("optimizer");
    paramsel->addOpt("W", rls->getOpt("W"));
    rls->removeOpt("W", false);

    delete tmp;
    delete[] lambdas;

    return paramsel;
}


}

#endif // _GURLS_CALIBRATESGD_H_
