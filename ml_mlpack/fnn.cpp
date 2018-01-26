#include <mlpack/core.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/core/optimizers/cne/cne.hpp>
using namespace mlpack;
using namespace mlpack::ann;
using namespace mlpack::optimization;
int main()
{
 /*
   * Create the four cases for XOR with two variable
    
   *  Input    Output
   * 0 XOR 0  =  0
   * 1 XOR 1  =  0
   * 0 XOR 1  =  1
   * 1 XOR 0  =  1
   */
  arma::mat train("1,0,0,1;1,0,1,0");
  arma::mat labels("1,1,2,2");
  // Network with 2 input nodes, 2 hidden nodes, and 2 output layer nodes.
  FFN<NegativeLogLikelihood<> > network;
  network.Add<Linear<> >(2, 2);
  network.Add<SigmoidLayer<> >();
  network.Add<Linear<> >(2, 2);
  network.Add<LogSoftMax<> >();
  // CNE object.
  CNE opt(20, 5000, 0.1, 0.02, 0.2, 0, 0);
  // Train the network with CNE.
  network.Train(train, labels, opt);
  // Predict for the same train data.
  arma::mat predictionTemp;
  network.Predict(train, predictionTemp);
  arma::mat prediction = arma::zeros<arma::mat>(1, predictionTemp.n_cols);
  for (size_t i = 0; i < predictionTemp.n_cols; ++i)
  {
    prediction(i) = arma::as_scalar(arma::find(
        arma::max(predictionTemp.col(i)) == predictionTemp.col(i), 1)) + 1;
  }
  // Print the results.

  for(size_t i = 0; i < 4; i++)
    std::cout << prediction << std::endl;
}