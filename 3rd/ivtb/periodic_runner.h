/*
 * Created by Ivan B on 2022/8/10.
 */

#ifndef IVTB_TOOLS_PERIODIC_RUNNER_H_
#define IVTB_TOOLS_PERIODIC_RUNNER_H_

#include <algorithm>
#include "stopwatch.h"

// todo ivan. consider these macros return whether executed this time?

// One level of macro indirection is required in order to resolve __LINE__,
// and get varname1 instead of varname__LINE__.
#define UNIQUE_NAME_CONCAT_VAR_INTERNAL(a,b) a##b
#define UNIQUE_NAME_CONCAT_VAR(a, b) UNIQUE_NAME_CONCAT_VAR_INTERNAL(a,b)
#define UNIQUE_NAME(base) UNIQUE_NAME_CONCAT_VAR(base, __LINE__)

#define RUN_EVERY_N(x, execution) static ivtb::NthRunner UNIQUE_NAME(runner_)(x); \
    if (UNIQUE_NAME(runner_).shouldRun()) {execution;}

#define RUN_N_TIMES(x, execution) static ivtb::FirstNRunner UNIQUE_NAME(runner_)(x); \
    if (UNIQUE_NAME(runner_).shouldRun()) {execution;}

#define RUN_PERIODICALLY(seconds, execution) static ivtb::PeriodicRunner UNIQUE_NAME(runner_)(seconds); \
    if (UNIQUE_NAME(runner_).shouldRun()) {execution;}

namespace ivtb{

class NthRunner {
    const unsigned int kMaxN;
    unsigned int n_;

  public:
    explicit NthRunner(unsigned int n): kMaxN(std::max(n,1u)){
        // so the first run would shoot
        n_ = kMaxN-1;
    }

    bool shouldRun(){
        if (++n_ == kMaxN){
            n_ = 0;
            return true;
        } else{
            return false;
        }
    }

};

class FirstNRunner {
    const unsigned int kMaxN;
    unsigned int n_=0;
  public:
    explicit FirstNRunner(unsigned int n): kMaxN(n){}

    bool shouldRun(){
        if (n_ >= kMaxN) return false;
        ++n_;
        return true;
    }
};

class PeriodicRunner {
    const double kPeriodSeconds;
    StopwatchMono stopwatch_{true};
  public:
    explicit PeriodicRunner(double period): kPeriodSeconds(period){}

    bool shouldRun(){
        if (stopwatch_.isPaused() || stopwatch_.passedSeconds() > kPeriodSeconds){
            stopwatch_.start();
            return true;
        }
        return false;
    }
};

}

#endif //IVTB_TOOLS_PERIODIC_RUNNER_H_
