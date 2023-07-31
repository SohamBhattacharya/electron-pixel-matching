# ifndef TreeOutputInfo_H
# define TreeOutputInfo_H


#include <any>
#include <cassert>
#include <iostream>
#include <map>
#include <stdlib.h>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>
#include <unordered_map>

#include <boost/type_index.hpp>

#include <TH1F.h>
#include <TH2F.h>
#include <TMatrixD.h>
#include <TROOT.h>
#include <TTree.h> 
#include <TVectorD.h> 

#include "ElectronPixelMatching/MyModules/interface/Common.h"
#include "ElectronPixelMatching/MyModules/interface/Constants.h"

template<typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec) {
    os << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        os << vec[i];
        if (i != vec.size() - 1)
            os << ", ";
    }
    os << "]";
    return os;
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<std::vector<T>>& vec) {
    os << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << vec[i] << ", ";
    }
    os << "]";
    return os;
}


namespace TreeOutputInfo
{
    class TreeOutput
    {
        public :
        
        TTree *tree;
        
        std::unordered_map <
            std::string,
            std::variant<
                int,
                ULong64_t,
                float,
                Common::vint,
                Common::vvint,
                Common::vfloat,
                Common::vvfloat
            >
        > m_content;
        
        template<typename T>
        void registerVar(std::string varName)
        {
            m_content[varName] = T();
            
            tree->Branch(varName.c_str(), &std::get<T>(m_content[varName]));
        }
        
        template<typename T>
        void setVar(const std::string& name, const T& value)
        {
            //printf("Trying setVar with \"%s\"<%s>.\n", name.c_str(), boost::typeindex::type_id<T>().pretty_name().c_str());
            assert(m_content.find(name) != m_content.end());
            //static_assert(std::is_same_v<std::decay_t<decltype(m_content[name])>, T> == true);
            //assert(constexpr(std::is_same_v<std::decay_t<decltype(m_content[name])>, T>));
            m_content[name] = value;
        }
        
        template<typename T>
        void fillVarV(const std::string& name, const T& value)
        {
            //printf("Trying fillVarV with \"%s\"<%s>.\n", name.c_str(), boost::typeindex::type_id<T>().pretty_name().c_str());
            assert(m_content.find(name) != m_content.end());
            auto& content = std::get<std::vector<T>>(m_content[name]);
            content.push_back(value);
        }
        
        template<typename T>
        void fillVarVV(const std::string& name, const std::vector<T>& value)
        {
            //printf("Trying fillVarVV with \"%s\"<%s>.\n", name.c_str(), boost::typeindex::type_id<T>().pretty_name().c_str());
            assert(m_content.find(name) != m_content.end());
            auto& content = std::get<std::vector<std::vector<T>>>(m_content[name]);
            content.push_back(value);
        }
        
        void clearVars()
        {
            for (auto& [ key, content ] : m_content)
            {
                std::visit([&key](auto&& arg)
                {
                    using TARG = std::decay_t<decltype(arg)>;
                    
                    if constexpr (
                        std::is_same_v<TARG, ULong64_t>
                        || std::is_same_v<TARG, int>
                        || std::is_same_v<TARG, float>
                    )
                    {
                        //std::cout << "clearVars(...): " << key << "\n";
                        arg = 0;
                    }
                    else if constexpr (
                        std::is_same_v<TARG, Common::vint>
                        || std::is_same_v<TARG, Common::vfloat>
                        || std::is_same_v<TARG, Common::vvint>
                        || std::is_same_v<TARG, Common::vvfloat>
                    )
                    {
                        //std::cout << "clearVars(...): " << key << "\n";
                        arg.clear();
                    }
                    else
                    {
                        static_assert(Common::always_false_v<TARG>, "non-exhaustive visitor!");
                    }
                }, content);
            }
        }
        
        void fill()
        {
            tree->Fill();
        }
        
        TreeOutput(std::string details, edm::Service<TFileService> fs)
        {
            //printf("Loading custom ROOT dictionaries. \n");
            //gROOT->ProcessLine(".L EDAnalyzers/TreeMaker/interface/CustomRootDict.cc+");
            //printf("Loaded custom ROOT dictionaries. \n");
            
            tree = fs->make<TTree>(details.c_str(), details.c_str());
        }
    };
    
    
    template<>
    void TreeOutput::setVar<double>(const std::string& name, const double& value)
    {
        TreeOutput::setVar<float>(name, (float) value);
    }
    
    template<>
    void TreeOutput::fillVarV<double>(const std::string& name, const double& value)
    {
        TreeOutput::fillVarV<float>(name, (float) value);
    }
}


# endif
