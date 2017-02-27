
# pylint: disable=E0001

import mnist_loader
import network
import sys
import datetime

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

def experiment(eta, mb_size, epochs, layers):
    print
    print "num hidden layers :", len(layers[1:-1]) 
    print "layers            :", layers
    print "learning rate     :", eta
    print "mini-batch size   :", mb_size
    print "epochs            :", epochs
    print 
    print "#"*40

    sys.stdout.flush()

    # net = network.Network([784, hidden_neurons, 10])
    net = network.Network(layers)
    net.SGD(training_data, epochs, mb_size, eta, test_data=test_data)


if __name__ == '__main__':
    epochs, mb_size, eta, layers = sys.argv[1:]
    eta, mb_size, epochs = float(eta), int(mb_size), int(epochs)
    fname = "experiments/e{}_m{}_r{}_l{}.txt".format( \
        epochs, mb_size, eta, layers)
    layers = [int(l) for l in layers.split('-')]
    sys.stdout=open(fname, 'w')
    print "#", fname[12:]
    print datetime.datetime.now()
    experiment(eta, mb_size, epochs, layers)
    sys.stdout.close()

# python2.7 experiment.py 30 10 100.0 784-30-10