#ifndef PID_H
#define PID_H
#include <vector>

class PID {
 public:
  /**
   * Constructor
   */
  PID();

  /**
   * Destructor.
   */
  virtual ~PID();

  /**
   * Initialize PID.
   * @param (Kp_, Ki_, Kd_, p_error_, i_error_. d_error_) The initial PID coefficients
   */
  void Init(double Kp_, double Ki_, double Kd_, double p_error_, double i_error_, double d_error_, int tuned_);

  /**
   * Update the PID error variables given cross track error.
   * @param cte The current cross track error
   */
  void UpdateError(double cte);

  /**
   * Calculate the total PID error.
   * @output The total PID error
   */
  double TotalError();

  //apply SGD to adjust PID controller's coefficients
  void SGD(double error,double alpha);

  //setter and getter of variable tunned
  void SetTunned(int flag);
  int GetTunned();

 private:
  /**
   * PID Errors
   */
  double p_error;
  double i_error;
  double d_error;

  /*
   * PID Coefficients
   */ 
  double Kp;
  double Ki;
  double Kd;

  //a vector to store i_error_buffer
  std::vector <double> i_error_buffer;

  //a flag indicates whether this controller is tuned or not
  int tuned;
};

#endif  // PID_H
