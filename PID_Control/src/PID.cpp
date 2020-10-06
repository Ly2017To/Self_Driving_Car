#include "PID.h"
#include <iostream>
#include <numeric>
using namespace std;

/**
 * TODO: Complete the PID class. You may add any additional desired functions.
 */

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp_, double Ki_, double Kd_, double p_error_, double i_error_, double d_error_, int tuned_) {
  /**
   * TODO: Initialize PID coefficients (and errors, if needed)
   */

  Kp=Kp_;
  Ki=Ki_;
  Kd=Kd_;
  p_error=p_error_;
  i_error=i_error_;
  d_error=d_error_;
  tuned=tuned_;
  
  return;
}

void PID::UpdateError(double cte) {
  /**
   * TODO: Update PID errors based on cte.
   */
  
  if(i_error_buffer.empty()==1||i_error_buffer.size()<10){
    i_error_buffer.push_back(cte);
  }else{
    i_error_buffer.erase(i_error_buffer.begin());
  }
  
  //i_error+=cte;
  i_error=accumulate(i_error_buffer.begin(), i_error_buffer.end(), 0.0);
  d_error=cte-p_error; //in this way, we do not have to create a variable the previous cte
  p_error=cte;

  return;
}

double PID::TotalError() {
  /**
   * TODO: Calculate and return the total error
   */
  
  return -Kp*p_error-Ki*i_error-Kd*d_error;  // TODO: Add your total error calc here!
}

//apply SGD to optimize the squre of total error
void PID::SGD(double error, double alpha){

   Kp=Kp+alpha*(error)*p_error;
   Ki=Ki+alpha*(error)*i_error;
   Kd=Kd+alpha*(error)*d_error;
   
   cout<<"p_error="<<p_error<<endl;
   cout<<"i_error="<<i_error<<endl;
   cout<<"d_error="<<d_error<<endl;
   cout<<"Kp="<<Kp<<" "<<"Ki="<<Ki<<" "<<"Kd="<<Kd<<" "<<endl;

   return;
}

void PID::SetTunned(int flag){
   tuned=flag;
}

int PID::GetTunned(){
   return tuned;
} 
