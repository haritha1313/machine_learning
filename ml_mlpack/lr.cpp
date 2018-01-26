#include <mlpack/core.hpp>
#include <mlpack/methods/logistic_regression/logistic_regression.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/core/optimizers/cne/cne.hpp>
using namespace std;
using namespace arma;
using namespace mlpack;
using namespace mlpack::ann;
using namespace mlpack::optimization;
using namespace mlpack::optimization::test;
using namespace mlpack::distribution;
using namespace mlpack::regression;
int main()
{
  // Generate a two-Gaussian dataset.
  GaussianDistribution g1(arma::vec("1.0 1.0 1.0"), arma::eye<arma::mat>(3, 3));
  GaussianDistribution g2(arma::vec("9.0 9.0 9.0"), arma::eye<arma::mat>(3, 3));
  arma::mat data(3, 1000);
  arma::Row<size_t> responses(1000);
  for (size_t i = 0; i < 500; ++i)
  {
    data.col(i) = g1.Random();
    responses[i] = 0;
  }
  for (size_t i = 500; i < 1000; ++i)
  {
    data.col(i) = g2.Random();
    responses[i] = 1;
  }
  // Shuffle the dataset.
  arma::uvec indices = arma::shuffle(arma::linspace<arma::uvec>(0,
      data.n_cols - 1, data.n_cols));
  arma::mat shuffledData(3, 1000);
  arma::Row<size_t> shuffledResponses(1000);
  for (size_t i = 0; i < data.n_cols; ++i)
  {
    shuffledData.col(i) = data.col(indices[i]);
    shuffledResponses[i] = responses[indices[i]];
  }
  // Create a test set.
  arma::mat testData(3, 1000);
  arma::Row<size_t> testResponses(1000);
  for (size_t i = 0; i < 500; ++i)
  {
    testData.col(i) = g1.Random();
    testResponses[i] = 0;
  }
  for (size_t i = 500; i < 1000; ++i)
  {
    testData.col(i) = g2.Random();
    testResponses[i] = 1;
  }
  // *******************************************************************
  CNE opt(50, 2000, 0.1, 0.02, 0.2, 1, 0);
  LogisticRegression<> lr(shuffledData, shuffledResponses, opt, 0.5);
  // *******************************************************************
  // Ensure that the error is close to zero. This is 100% means no error
  const double acc = lr.ComputeAccuracy(data, responses);
  cout << acc << endl;
  // Check if optimization happened correctly or not by using test set.
  const double testAcc = lr.ComputeAccuracy(testData, testResponses);
  // 100% means no error.
  cout << testAcc << endl;
}