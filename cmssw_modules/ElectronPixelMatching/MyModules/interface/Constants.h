# ifndef Constants_H
# define Constants_H


#include <iostream>
#include <map>
#include <stdlib.h>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <TH1F.h>
#include <TH2F.h>
#include <TMatrixD.h>
#include <TTree.h> 
#include <TVectorD.h> 


namespace Constants
{
    double HGCAL_MINETA = 1.4; //1.479
    double HGCAL_MAXETA = 3.2;
    
    const int HGCEE_NLAYER = 28;
    
    const double DEFAULT_LARGE_DR = 999;
    
}


# endif
