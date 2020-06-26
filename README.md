The purpose of this repo is to compare feed forward neural networks, arima
methods, and convolutional neural networks for regression performance on
NE-ISO electricity pricing data set.

usage: nn.py [-h] [-e EPOCHS] [-o OPTIMIZE] [-m MODEL] [-s SPAN] [-v VARIANCE] [-w WRITE] data

positional arguments:
  data                  data file

optional arguments:
  -h, --help            show this help message and exit
  -e EPOCHS, --epochs EPOCHS
                        number of epochs
  -o OPTIMIZE, --optimize OPTIMIZE
                        number of samples
  -m MODEL, --model MODEL
                        model file
  -s SPAN, --span SPAN  date range
  -v VARIANCE, --variance VARIANCE
                        pca variance
  -w WRITE, --write WRITE
                        write results

usage: arima.py [-h] [-m MODEL] [-s SPAN] [-v VARIANCE] [-w WRITE] data

positional arguments:
  data                  data file

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        model file
  -s SPAN, --span SPAN  date range
  -v VARIANCE, --variance VARIANCE
                        pca variance
  -w WRITE, --write WRITE
                        write results

