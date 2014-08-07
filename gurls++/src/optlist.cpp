/*
  * The GURLS Package in C++
  *
  * Copyright (C) 2011-1013, IIT@MIT Lab
  * All rights reserved.
  *
 * author:  M. Santoro
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

#include <string>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <vector>

#include "gurls++/options.h"
#include "gurls++/optfunction.h"
#include "gurls++/optlist.h"
#ifdef _BGURLS
#include "bgurls++/bigarray.h"
#endif
#include "gurls++/serialization.h"

#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>

using namespace std;

namespace gurls{

void GurlsOptionsList::setName(std::string newname)
{
    name = newname;
    removeOpt("Name");
    (*table)["Name"] = new OptString(newname);
}

GurlsOptionsList::GurlsOptionsList(std::string ExpName, bool usedefopt): GurlsOption(OptListOption), name(ExpName)
{
    table = new std::map<std::string, GurlsOption* >();

    (*table)["Name"] = new OptString(ExpName);

    if(usedefopt)
    {

        //		opt.combineclasses = @mean; % How to combine performance measure per class (mean/median/min/max?)
        (*table)["combineclasses"] = new OptFunction("mean");

        (*table)["name"] = new OptString(ExpName);
        (*table)["plotstr"] = new OptString(ExpName);

#ifdef USE_BINARY_ARCHIVES
        (*table)["savefile"] = new OptString(ExpName.append(".bin"));
#else
        (*table)["savefile"] = new OptString(ExpName.append(".txt"));
#endif

        // ================================================== Algorithm options

        //		opt.kernel.type = 'rbf';
        (*table)["singlelambda"] = new OptFunction("median");
        (*table)["predbagmethod"] = new OptString("vote");

        // NOTE: lambda is searched between
        // [min(eig_r, opt.smallnumber), eig_1],
        // where r = rank, eig_1 = max eig val.
        (*table)["smallnumber"] = new OptNumber(1e-8);

        // ================================================== Directory options
        (*table)["tmpdir"] = new OptString(ExpName);

        // ===================================================== Output options
        (*table)["savekernel"] = new OptNumber(1);
        (*table)["saveanalysis"] = new OptNumber(1);
        //		opt.hoperf = @perf_precrec;
        (*table)["ploteval"] = new OptString("acc");
        //		WARNING: this should be an array of strings...
        (*table)["perfeval"] = new OptString("acc");

        // ======================================================== Data option
        (*table)["nholdouts"] = new OptNumber(1);
        (*table)["hoproportion"] = new OptNumber(0.2);
        (*table)["hoperf"] = new OptString("macroavg");
//        (*table)["nlambda"] = new OptNumber(100);
        (*table)["nsigma"] =  new OptNumber(25);
        (*table)["nlambda"] = new OptNumber(20);
//        (*table)["nsigma"] =  new OptNumber(10);
        (*table)["eig_percentage"] = new OptNumber(5);


    // ======================================================== Pegasos option
        (*table)["subsize"]   = new OptNumber(50);
        (*table)["calibfile"] = new OptString("foo");
        (*table)["epochs"]   = new OptNumber(4);

        // ============================================================== Quiet
        // Currenty either 0 or 1; levels of verbosity may be implemented later;
        (*table)["verbose"] = new OptNumber(1);

        // ======================================================= Version info
        (*table)["version"] = new OptString("2.0");


        GurlsOptionsList * randfeats = new GurlsOptionsList("randfeats");
        randfeats->table->insert(pair<std::string,GurlsOption*>("D", new OptNumber(500)));
        randfeats->table->insert(pair<std::string,GurlsOption*>("samplesize", new OptNumber(100)));

        (*table)["randfeats"] = randfeats;

    }

}

GurlsOptionsList::GurlsOptionsList(const GurlsOptionsList &other): GurlsOption(OptListOption)
{
    table = new ValueType();

    ValueType::const_iterator it, end;

    for(it = other.table->begin(), end = other.table->end(); it != end; ++it)
        copyOpt(it->first, other);

    name = getOptAsString("Name");
}

GurlsOptionsList::~GurlsOptionsList()
{
    ValueType::iterator it, end;

    for(it = table->begin(), end = table->end(); it != end; ++it)
        delete (it->second);

    table->clear();
    delete table;
}

void GurlsOptionsList::printAll()
{
    std::cout << *this;
}

bool GurlsOptionsList::hasOpt(string key) const
{
    //return table->count(key)>0;
    try
    {
        getOpt((key));
        return true;
    }
    catch(gException&)
    {
        return false;
    }
}

void GurlsOptionsList::removeOpt(string key, bool deleteMembers)
{
    ValueType::iterator it = table->find(key);
    if(it != table->end())
    {
        if (deleteMembers)
            delete it->second;

        table->erase(it);
    }
}

bool GurlsOptionsList::isA(OptTypes id) const
{
    return (id == OptListOption);
}

GurlsOptionsList *GurlsOptionsList::dynacast(GurlsOption *opt)
{
    if (opt->isA(OptListOption) )
       return static_cast<GurlsOptionsList*>(opt);

    throw gException(gurls::Exception_Illegal_Dynamic_Cast);
}

const GurlsOptionsList *GurlsOptionsList::dynacast(const GurlsOption *opt)
{
    if (opt->isA(OptListOption) )
        return static_cast<const GurlsOptionsList*>(opt);

    throw gException(gurls::Exception_Illegal_Dynamic_Cast);
}

int GurlsOptionsList::size() const
{
    return table->size();
}

GurlsOption *GurlsOptionsList::operator [](int idx)
{
    if ( idx >= this->size() )
        throw gException(gurls::Exception_Index_Out_of_Bound);

    ValueType::iterator itr = table->begin();

    for (int i = 0; i<idx; ++i, ++itr){}    // Do nothing else then following the iterator

    return itr->second;
}

std::ostream& GurlsOptionsList::operator<<(std::ostream& os) const
{
    return os << *this;
}

string GurlsOptionsList::toString()
{
    std::stringstream stream;
    stream << (*this);

    return stream.str();
}

//template<typename T>
//GurlsOption* copyOptMatrix(const GurlsOption* toCopy)
//{
//    const gMat2D<T> & mat = OptMatrix<gMat2D<T> >::dynacast(toCopy)->getValue();

//    gMat2D<T>* newMat = new gMat2D<T>(mat);
//    return new OptMatrix<gMat2D<T> >(*newMat);
//}

template<class MatrixType>
GurlsOption* copyOptMatrix(const GurlsOption* toCopy)
{
    const MatrixType & mat = OptMatrix<MatrixType>::dynacast(toCopy)->getValue();

    MatrixType* newMat = new MatrixType(mat);
    return new OptMatrix<MatrixType >(*newMat);
}

void GurlsOptionsList::copyOpt(string key, const GurlsOptionsList &from)
{

    const GurlsOption* toCopy = from.getOpt(key);

    GurlsOption* newOpt = NULL;

    switch(toCopy->getType())
    {
    case StringOption:
        newOpt = new OptString(OptString::dynacast(toCopy)->getValue());
        break;
    case NumberOption:
        newOpt = new OptNumber(OptNumber::dynacast(toCopy)->getValue());
        break;
    case StringListOption:
        newOpt = new OptStringList(OptStringList::dynacast(toCopy)->getValue());
        break;
    case NumberListOption:
        newOpt = new OptNumberList(OptNumberList::dynacast(toCopy)->getValue());
        break;
    case FunctionOption:
        newOpt = new OptFunction(OptFunction::dynacast(toCopy)->getName());
        break;
    case ProcessOption:
        newOpt = new OptProcess(*OptProcess::dynacast(toCopy));
        break;
    case MatrixOption:
    case VectorOption:
    {
       const OptMatrixBase* base = dynamic_cast<const OptMatrixBase*>(toCopy);

        if(base == NULL)
            throw gException(Exception_Illegal_Dynamic_Cast);

#ifdef _BGURLS
        if(base->hasBigArray())
        {
            switch(base->getMatrixType())
            {
                case OptMatrixBase::ULONG:
                    newOpt = copyOptMatrix<BigArray<unsigned long> >(toCopy);
                    break;
                case OptMatrixBase::FLOAT:
                    newOpt = copyOptMatrix<BigArray<float> >(toCopy);
                    break;
                case OptMatrixBase::DOUBLE:
                    newOpt = copyOptMatrix<BigArray<double> >(toCopy);
                    break;
            }
        }
        else
#endif
        {
            switch(base->getMatrixType())
            {
                case OptMatrixBase::ULONG:
                    newOpt = copyOptMatrix<gMat2D<unsigned long> >(toCopy);
                    break;
                case OptMatrixBase::FLOAT:
                    newOpt = copyOptMatrix<gMat2D<float> >(toCopy);
                    break;
                case OptMatrixBase::DOUBLE:
                    newOpt = copyOptMatrix<gMat2D<double> >(toCopy);
                    break;
            }
        }

    }
        break;
    case OptListOption:
    {
        const GurlsOptionsList* toCopy_list = GurlsOptionsList::dynacast(toCopy);

        newOpt = new GurlsOptionsList(*toCopy_list);
    }
        break;
    case OptArrayOption:
        newOpt = new OptArray(*OptArray::dynacast(toCopy));
        break;
    case TaskSequenceOption:
        newOpt = new OptTaskSequence(OptTaskSequence::dynacast(toCopy)->getValue());
        break;
    case TaskIDOption:
    case GenericOption:
        break;
    }

    if(newOpt != NULL)
        addOpt(key, newOpt);
}

/**
  * Writes a GurlsOptionsList to a stream
  */
GURLS_EXPORT std::ostream& operator<<(std::ostream& os, const GurlsOptionsList& opt)
{
    std::map<std::string, GurlsOption* >::iterator it;

    os << std::endl << "~~~~~~~ GurlsOptionList: " << opt.getName() << std::endl;

    for (it = opt.table->begin(); it != opt.table->end(); ++it)
        os << "\t[ " << it->first << " ] = " << *(it->second) << endl;

    os << "~~~~~~~";
    return os;
}

bool GurlsOptionsList::addOpt(std::string key, GurlsOption* value)
{
    std::pair<ValueType::iterator, bool> res = table->insert( pair<std::string,GurlsOption*>(key, value) );

    if(!res.second)
        throw gException(Exception_Parameter_Already_Definied + " (" + key + ")");

    return true;
}

bool GurlsOptionsList::addOpt(std::string key, std::wstring value)
{
    std::string val = std::string(value.begin(), value.end());
    return addOpt(key, val);
}

bool GurlsOptionsList::addOpt(std::string key, std::string value)
{
    OptString* v = new OptString(value);
    try
    {
        return addOpt(key, v);
    }
    catch (gException & ex)
    {
        delete v;
        throw ex;
    }
}

GurlsOption* GurlsOptionsList::getOpt(std::string key)
{
    if(key.empty())
        throw gException(Exception_Parameter_Not_Definied_Yet + "( )");

    std::vector<std::string> names;
    boost::split(names, key, boost::is_any_of("."));

    GurlsOption* gout = this;
    ValueType *tab;

    for(std::vector<std::string>::iterator n_it = names.begin(); n_it != names.end(); ++n_it)
    {
        tab = GurlsOptionsList::dynacast(gout)->table;

        std::map<std::string, GurlsOption* >::iterator it = tab->find(*n_it);

        if(it == tab->end())
            throw gException(Exception_Parameter_Not_Definied_Yet + "( " + *n_it + " )");

        gout = it->second;
    }

    return gout;
}

const GurlsOption* GurlsOptionsList::getOpt(std::string key) const
{
    if(key.empty())
        throw gException(Exception_Parameter_Not_Definied_Yet + "( )");

    std::vector<std::string> names;
    boost::split(names, key, boost::is_any_of("."));

    const GurlsOption* gout = this;
    ValueType *tab;

    for(std::vector<std::string>::iterator n_it = names.begin(); n_it != names.end(); ++n_it)
    {
        tab = GurlsOptionsList::dynacast(gout)->table;

        std::map<std::string, GurlsOption* >::iterator it = tab->find(*n_it);

        if(it == tab->end())
            throw gException(Exception_Parameter_Not_Definied_Yet + "( " + *n_it + " )");

        gout = it->second;
    }

    return gout;
}

std::string GurlsOptionsList::getOptAsString(std::string key) const
{
    return getOptValue<OptString>(key);
}

string GurlsOptionsList::getName() const
{
    return this->name;
}

const GurlsOptionsList::ValueType &GurlsOptionsList::getValue() const
{
    return *table;
}

double GurlsOptionsList::getOptAsNumber(std::string key) const
{
    return getOptValue<OptNumber>(key);
}

void GurlsOptionsList::save(const std::string& fileName) const
{
#ifndef USE_BINARY_ARCHIVES
    std::ofstream outstream(fileName.c_str());
#else
    std::ofstream outstream(fileName.c_str(), ios_base::binary);
#endif

    if(!outstream.is_open())
        throw gException("Could not open file " + fileName);

    oarchive outar(outstream);
    outar << *this;

    outstream.close();
}

void GurlsOptionsList::load(const std::string& fileName)
{
#ifndef USE_BINARY_ARCHIVES
    std::ifstream instream(fileName.c_str());
#else
    std::ifstream instream(fileName.c_str(), ios_base::binary);
#endif

    if(!instream.is_open())
        throw gException("Could not open file " + fileName);

    try
    {
        iarchive inar(instream);
        inar >> *this;
    }
    catch(boost::archive::archive_exception&)
    {
        instream.close();
        throw gException("Invalid file format for " + fileName);
    }

    instream.close();
}


}

