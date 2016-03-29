#include <iostream>
using namespace std;
class CHelper{
public:
    CHelper(){}
    float getHelpNumber(char* string){
        return ((float *)string)[0];
    }
};

extern "C"{
    CHelper* helper(){ return new CHelper();}
    float getHelpNumber(CHelper *helper, char* string){
        return helper->getHelpNumber(string);
    }
}
