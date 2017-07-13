#include "alg_detecting.h"

DetectHead::DetectHead()
{
}

DetectHead::pDetectHead = new DetectHead;
DetectHead::pThread = NULL;

DetectHead *DetectHead::GetInstance()
{
    return pDetectHead;
}

int DetectHead::Start()
{
    if (pThread)
    {
        return ALG_ALREAD_START;
    }
    pThread = new std::thread();
    return SUCCEED;
}

int DetectHead::Stop()
{
    
}
